import os
import io
import time
import shutil
import sqlite3
import asyncio
from credentials import URL, INCREMENTAL_URL

import aiohttp
import requests
import numpy as np
import pandas as pd


LOCAL_DB_DEFAULT = "data/tokens_raw.db"
LOCAL_DB_CANDIDATES = [
    os.getenv("TOKENS_LOCAL_DB", LOCAL_DB_DEFAULT),
    "data/tokens_raw.db",
]

# --- PARAMETRY API DEXSCREENER -----------------------------------------
# Oficjalny limit dla /latest/dex/pairs: 300 req/min
# https://docs.dexscreener.com/api/reference  (rate-limit 300 requests per minute)
DEXSCREENER_MAX_REQUESTS_PER_MIN = 300
DEXSCREENER_MARGIN = 0.98  # zostawiamy sobie ~2% marginesu
DEXSCREENER_REQUESTS_PER_SECOND = (
    DEXSCREENER_MAX_REQUESTS_PER_MIN / 60.0 * DEXSCREENER_MARGIN
)

CHUNK_SIZE = 30  # max 30 par w jednym request
CONCURRENT_REQUESTS = 20  # ile requestów naraz może wisieć (pipelining)


# =======================================================================
# Rate limiter – płynny (brak 60-sekundowych „drzemek”)
# =======================================================================
class RateLimiter:
    """
    Prosty globalny limiter:
    - pilnuje średnio ~DEXSCREENER_REQUESTS_PER_SECOND requestów na sekundę
    - działa na zasadzie: między kolejnymi requestami jest >= 1/rate sekundy
    - przy concurrency 20 i tak będzie globalnie ~4.9 req/s (ok. 294/min)
    """

    def __init__(self, rate_per_sec: float):
        self.rate = float(rate_per_sec)
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if now < self._next_allowed:
                await asyncio.sleep(self._next_allowed - now)
                now = time.monotonic()
            interval = 1.0 / self.rate
            # następny dopuszczalny czas = max(obecny, poprzedni) + interval
            self._next_allowed = max(self._next_allowed, now) + interval


# =======================================================================
# Helpers
# =======================================================================
def chunked(iterable, chunk_size):
    """Generator zwracający kolejne listy indeksów o rozmiarze chunk_size."""
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i : i + chunk_size]


async def update_worthy(df: pd.DataFrame, output_filename: str) -> None:
    """
    Uzupełnia current_* oraz meta-fiely (website, twitter, telegram, symbol, name)
    TYLKO tam, gdzie brakuje danych (NaN / "").

    Optymalizacje:
    - płynny rate-limiter na ~294 req/min (lekko poniżej 300/min)
    - zapis CSV chunkami (30 wierszy na raz) zamiast per wiersz
    - iteracja po indeksach (df.at) zamiast iterrows()
    """

    session_timeout = aiohttp.ClientTimeout(total=30)
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    file_lock = asyncio.Lock()
    rate_limiter = RateLimiter(DEXSCREENER_REQUESTS_PER_SECOND)

    max_attempts = 25
    initial_delay = 2.0

    # ---------- helper: extract meta fields from a single pair ----------
    def _extract_pair_meta(p: dict):
        info = p.get("info", {}) or {}
        websites = info.get("websites", []) or []
        website_url = websites[0].get("url", "") if websites else ""

        twitter_url = ""
        telegram_url = ""
        socials = info.get("socials", []) or []
        for s in socials:
            if s.get("type") == "twitter":
                twitter_url = s.get("url", "") or ""
            elif s.get("type") == "telegram":
                telegram_url = s.get("url", "") or ""

        base = p.get("baseToken", {}) or {}
        return (
            website_url,
            twitter_url,
            telegram_url,
            base.get("symbol", "") or "",
            base.get("name", "") or "",
        )

    # -------------------------------------------------------------------
    def deep_float(d: dict, *keys) -> float:
        for k in keys:
            if not isinstance(d, dict):
                return np.nan
            d = d.get(k)
            if d is None:
                return np.nan
        try:
            return float(d)
        except (TypeError, ValueError):
            return np.nan

    async def fetch_and_update_chunk(chunk_indices):
        nonlocal max_attempts, initial_delay

        # adresy par dla chunku
        addresses = df.loc[chunk_indices, "pairAddress"].astype(str).tolist()
        url = (
            "https://api.dexscreener.com/latest/dex/pairs/solana/"
            f"{','.join(addresses)}"
        )

        attempt = 0
        delay = initial_delay

        while attempt < max_attempts:
            try:
                async with semaphore:
                    await rate_limiter.acquire()
                    async with session.get(url) as response:
                        status = response.status

                        if status == 200:
                            data = await response.json()

                            pairs_raw = data.get("pairs")
                            if not pairs_raw:
                                # API nic nie zwróciło dla tego zestawu – ustawiamy NaN-y w current_*
                                for idx in chunk_indices:
                                    df.at[idx, "current_priceUsd"] = np.nan
                                    df.at[idx, "current_txns_h24_buy"] = np.nan
                                    df.at[idx, "current_txns_h24_sells"] = np.nan
                                    df.at[idx, "current_volume_h24"] = np.nan
                                    df.at[idx, "current_liquidity_usd"] = np.nan

                                chunk_df = df.loc[chunk_indices].copy()
                                async with file_lock:
                                    await asyncio.to_thread(
                                        chunk_df.to_csv,
                                        output_filename,
                                        mode="a",
                                        header=False,
                                        index=False,
                                    )
                                return

                            pairs_dict = {
                                (p.get("address") or p.get("pairAddress")): p
                                for p in pairs_raw
                                if (p.get("address") or p.get("pairAddress"))
                            }

                            # aktualizacja po indeksach, bez iterrows()
                            for idx in chunk_indices:
                                pair_address = str(df.at[idx, "pairAddress"])
                                p = pairs_dict.get(pair_address)

                                if p is None:
                                    df.at[idx, "current_priceUsd"] = np.nan
                                    df.at[idx, "current_txns_h24_buy"] = np.nan
                                    df.at[idx, "current_txns_h24_sells"] = np.nan
                                    df.at[idx, "current_volume_h24"] = np.nan
                                    df.at[idx, "current_liquidity_usd"] = np.nan
                                else:
                                    # ---- current_* ----
                                    df.at[idx, "current_priceUsd"] = deep_float(
                                        p, "priceUsd"
                                    )
                                    df.at[idx, "current_txns_h24_buy"] = deep_float(
                                        p, "txns", "h24", "buys"
                                    )
                                    df.at[idx, "current_txns_h24_sells"] = deep_float(
                                        p, "txns", "h24", "sells"
                                    )
                                    df.at[idx, "current_volume_h24"] = deep_float(
                                        p, "volume", "h24"
                                    )
                                    df.at[idx, "current_liquidity_usd"] = deep_float(
                                        p, "liquidity", "usd"
                                    )

                                    # ---- meta fields ----
                                    (
                                        website_url,
                                        twitter_url,
                                        telegram_url,
                                        sym,
                                        name,
                                    ) = _extract_pair_meta(p)

                                    def _missing(col_name: str) -> bool:
                                        v = df.at[idx, col_name]
                                        return pd.isna(v) or v == ""

                                    if _missing("website_url") and website_url:
                                        df.at[idx, "website_url"] = website_url
                                    if _missing("twitter_url") and twitter_url:
                                        df.at[idx, "twitter_url"] = twitter_url
                                    if _missing("telegram_url") and telegram_url:
                                        df.at[idx, "telegram_url"] = telegram_url
                                    if _missing("baseTokenSymbol") and sym:
                                        df.at[idx, "baseTokenSymbol"] = sym
                                    if _missing("baseTokenName") and name:
                                        df.at[idx, "baseTokenName"] = name

                            # zapisujemy chunk raz
                            chunk_df = df.loc[chunk_indices].copy()
                            async with file_lock:
                                await asyncio.to_thread(
                                    chunk_df.to_csv,
                                    output_filename,
                                    mode="a",
                                    header=False,
                                    index=False,
                                )
                            return

                        elif status == 429:
                            # Too many requests – mimo limiter'a może się zdarzyć
                            retry_after_header = response.headers.get("Retry-After")
                            try:
                                retry_after = int(retry_after_header)
                            except (TypeError, ValueError):
                                retry_after = delay
                            await asyncio.sleep(retry_after + 1)
                            delay = min(delay * 2, 60)
                            attempt += 1
                        else:
                            # inne błędy tymczasowe
                            await asyncio.sleep(delay)
                            delay = min(delay * 2, 60)
                            attempt += 1

            except (aiohttp.ClientError, asyncio.TimeoutError):
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)
                attempt += 1

        # po wyczerpaniu prób: NaN w current_* + zapis chunku
        for idx in chunk_indices:
            df.at[idx, "current_priceUsd"] = np.nan
            df.at[idx, "current_txns_h24_buy"] = np.nan
            df.at[idx, "current_txns_h24_sells"] = np.nan
            df.at[idx, "current_volume_h24"] = np.nan
            df.at[idx, "current_liquidity_usd"] = np.nan

        chunk_df = df.loc[chunk_indices].copy()
        async with file_lock:
            await asyncio.to_thread(
                chunk_df.to_csv,
                output_filename,
                mode="a",
                header=False,
                index=False,
            )

    async with aiohttp.ClientSession(timeout=session_timeout) as session:
        indices = df.index.tolist()
        tasks = [
            asyncio.create_task(fetch_and_update_chunk(chunk_indices))
            for chunk_indices in chunked(indices, CHUNK_SIZE)
        ]
        await asyncio.gather(*tasks)


def main():
    # --- wybór lokalnej ścieżki do DB ---
    local_db = None
    for cand in LOCAL_DB_CANDIDATES:
        if cand and os.path.exists(cand):
            local_db = cand
            break
    if not local_db:
        local_db = LOCAL_DB_CANDIDATES[0]
        os.makedirs(os.path.dirname(local_db), exist_ok=True)

    # --- czy istnieje tabela tokens? ---
    table_exists = False
    if os.path.exists(local_db):
        with sqlite3.connect(local_db) as con:
            q = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='tokens';"
            ).fetchone()
            table_exists = q is not None

    if not table_exists:
        # pierwszy bootstrap: ściągamy pełną DB
        try:
            with requests.get(URL, stream=True, timeout=15) as r:
                r.raise_for_status()
                with open(local_db, "wb") as f:
                    shutil.copyfileobj(r.raw, f)
            table_exists = True
            print("Bootstrapped local DB from full download.")
        except requests.RequestException:
            raise FileNotFoundError(
                "Brak lokalnej DB z tabelą 'tokens' i brak możliwości bootstrapa. "
                "Udostępnij pełną DB pod URL lub wrzuć lokalny .db z tabelą 'tokens'."
            )

    # --- incremental update po max(pairCreatedAt) ---
    max_created_iso = None
    try:
        with sqlite3.connect(local_db) as con:
            row = con.execute("SELECT MAX(pairCreatedAt) FROM tokens;").fetchone()
            max_created_iso = row[0] if row and row[0] else None
    except sqlite3.DatabaseError as e:
        print(f"SQLite check failed ({e}); skipping incremental fetch.")

    if max_created_iso:
        try:
            params = {"created_after": str(max_created_iso)}
            r = requests.get(INCREMENTAL_URL, params=params, timeout=15)
            if r.status_code == 200 and r.content and len(r.content) > 0:
                df_new = pd.read_csv(io.StringIO(r.text))
                if not df_new.empty:
                    with sqlite3.connect(local_db) as con:
                        con.execute(
                            "CREATE UNIQUE INDEX IF NOT EXISTS idx_tokens_pairAddress "
                            "ON tokens(pairAddress);"
                        )
                        df_new.to_sql(
                            "tokens_new", con, if_exists="replace", index=False
                        )
                        con.execute(
                            "INSERT OR REPLACE INTO tokens SELECT * FROM tokens_new;"
                        )
                        con.execute("DROP TABLE tokens_new;")
                    print(f"Incremental update applied: +{len(df_new)} rows.")
                else:
                    print("Incremental: no new rows.")
            else:
                print(
                    f"Incremental fetch returned {r.status_code}; using local DB only."
                )
        except requests.RequestException as e:
            print(
                f"Incremental source not reachable ({type(e).__name__}); using local DB only."
            )
    else:
        print("No max(pairCreatedAt) found; skipping incremental fetch.")

    # --- wczytanie df z lokalnej DB ---
    t0 = time.time()
    print("Reading tokens table from SQLite...")
    with sqlite3.connect(local_db) as con:
        df = pd.read_sql_query("SELECT * FROM tokens;", con)
    print(
        f"Loaded tokens table: {len(df)} rows x {df.shape[1]} cols "
        f"in {time.time() - t0:.1f}s"
    )

    # opcjonalny pełny dump (dla debugowania)
    # domyślnie WYŁĄCZONY – włączysz ustawiając TOKENS_SKIP_FETCHED_DUMP=0
    t1 = time.time()
    if os.getenv("TOKENS_SKIP_FETCHED_DUMP", "1") != "1":
        os.makedirs("data", exist_ok=True)
        print("Dumping data/tokens_fetched_db.csv (can be large)...")
        df.to_csv("data/tokens_fetched_db.csv", index=False)
        print(f"CSV dump finished in {time.time() - t1:.1f}s")
    else:
        print("Skipping tokens_fetched_db.csv dump (TOKENS_SKIP_FETCHED_DUMP=1).")

    base_columns = df.columns.tolist()
    additional_columns = [
        "current_priceUsd",
        "current_txns_h24_buy",
        "current_txns_h24_sells",
        "current_volume_h24",
        "current_liquidity_usd",
    ]
    all_columns = base_columns + [
        col for col in additional_columns if col not in base_columns
    ]
    df = df.reindex(columns=all_columns)

    # --- filtering by quoteTokenAddress + stats ---
    TARGET_QUOTE = "So11111111111111111111111111111111111111112"

    total_rows_before = len(df)
    mask_target = df["quoteTokenAddress"] == TARGET_QUOTE
    dropped_wrong_quote = int((~mask_target).sum())

    df = df[mask_target].copy()

    if df.empty:
        print(
            f"After filtering quoteTokenAddress != {TARGET_QUOTE} "
            "no rows remain – exiting."
        )
        return

    rows_before_grouping = len(df)

    # --- duplicate stats by baseTokenAddress ---
    group_sizes = df.groupby("baseTokenAddress").size()
    dup_keys = group_sizes[group_sizes > 1].index
    rows_in_multi_groups = (
        int(group_sizes[group_sizes > 1].sum()) if len(dup_keys) > 0 else 0
    )

    print(
        f"baseTokenAddress groups: total_unique={len(group_sizes)}, "
        f"duplicate_groups={len(dup_keys)}, "
        f"rows_in_duplicate_groups={rows_in_multi_groups}."
    )

    if len(dup_keys) == 0:
        # No duplicates at all → skip heavy aggregation
        print(
            "No duplicate baseTokenAddress values – skipping aggregation, df unchanged."
        )
    else:
        # Split into singleton and duplicate groups
        dup_mask = df["baseTokenAddress"].isin(dup_keys)
        df_single = df[~dup_mask].copy()
        df_dup = df[dup_mask].copy()

        print(
            f"Aggregating {len(df_dup)} rows in duplicate groups "
            f"(keeping {len(df_single)} singleton rows unchanged)..."
        )

        group_key = ["baseTokenAddress"]

        numeric_cols = df_dup.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in group_key]
        non_numeric_cols = [
            c for c in df_dup.columns if c not in numeric_cols + group_key
        ]

        def _mode(series: pd.Series):
            mode = series.mode(dropna=True)
            if not mode.empty:
                return mode.iloc[0]
            # all NaN / empty
            return np.nan

        agg_dict = {col: "mean" for col in numeric_cols}
        agg_dict.update({col: _mode for col in non_numeric_cols})

        t2 = time.time()
        df_dup_agg = df_dup.groupby("baseTokenAddress", as_index=False).agg(agg_dict)
        print(
            f"Aggregation over duplicate groups finished in {time.time() - t2:.1f}s, "
            f"aggregated_shape={df_dup_agg.shape}"
        )

        # combine back: singletons + aggregated duplicates
        df = pd.concat([df_single, df_dup_agg], ignore_index=True)

    # restore target column order (stable CSV header)
    df = df.reindex(columns=all_columns)

    # --- diagnostics ---
    print(
        f"Dropped {dropped_wrong_quote} rows with quoteTokenAddress != {TARGET_QUOTE} "
        f"(total loaded: {total_rows_before})."
    )
    print(
        f"baseTokenAddress grouping summary: before={rows_before_grouping}, "
        f"after={len(df)}, "
        f"rows_in_duplicate_groups={rows_in_multi_groups}."
    )

    # --- cutoff by pair age ---
    df["pairCreatedAt"] = pd.to_datetime(df["pairCreatedAt"])
    cutoff = pd.Timestamp.now() - pd.Timedelta(hours=24)
    df = df[df["pairCreatedAt"] < cutoff]
    print(f"shape after cutoff & grouping: {df.shape}")

    # --- RESUME: pomijanie już przetworzonych wierszy (pairAddress) ---
    output_filename = "data/tokens_raw_reclassified.csv"
    if os.path.exists(output_filename):
        try:
            completed_addresses = pd.read_csv(
                output_filename, usecols=["pairAddress"], dtype={"pairAddress": str}
            )["pairAddress"]
            completed_set = set(completed_addresses.dropna().astype(str))
            before_count = len(df)
            df = df[~df["pairAddress"].astype(str).isin(completed_set)]
            print(
                f"Resuming: skipped {before_count - len(df)} rows already in CSV; remaining {len(df)}."
            )
        except Exception as e:
            print(
                f"Resume warning: could not read existing CSV ({e}); processing all rows."
            )

    # nagłówek CSV przy pierwszym uruchomieniu
    if not os.path.exists(output_filename):
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, "w", encoding="utf-8", newline="") as f:
            df.head(0).to_csv(f, index=False)

    if df.empty:
        print("No rows left to process – exiting.")
        return

    print(
        "Starting update of current_priceUsd, current_txns_h24_*, "
        "current_volume_h24 and current_liquidity_usd..."
    )
    asyncio.run(update_worthy(df, output_filename))
    print(f"Update completed, data saved to {output_filename}")


if __name__ == "__main__":
    main()
