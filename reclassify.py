import pandas as pd
import aiohttp
import asyncio
import requests, shutil, sqlite3
import numpy as np
import os
import io

URL = "http://35.208.181.252:8000/download-db"  # full DB (fallback only)
INCREMENTAL_URL = "http://35.208.181.252:8000/tokens-since"  # CSV, filter by ?created_after=ISO8601

LOCAL_DB_DEFAULT = "data/tokens_raw.db"
LOCAL_DB_CANDIDATES = [
    os.getenv("TOKENS_LOCAL_DB", LOCAL_DB_DEFAULT),
    r"C:\Cloud\filips19mail\github\solanaDEXtokenCollector\data\tokens_raw.db",
]


def chunked(iterable, chunk_size):
    """Generator returning subsets (chunks) of size 'chunk_size'."""
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i : i + chunk_size]

def write_row_to_csv(row, output_filename):
    """Function to save a single row to CSV in append mode."""
    row.to_frame().T.to_csv(output_filename, mode="a", header=False, index=False)


async def update_worthy(df, output_filename):
    """
    Update *current_* metrics and fill missing values in:
    website_url, twitter_url, telegram_url, baseTokenSymbol, baseTokenName
    (only where they are empty or NaN).
    """
    session_timeout = aiohttp.ClientTimeout(total=30)
    semaphore       = asyncio.Semaphore(5)
    lock            = asyncio.Lock()

    max_attempts  = 10
    initial_delay = 2

    # ---------- helper: extract meta fields from a single pair ----------
    def _extract_pair_meta(p: dict) -> tuple[str, str, str, str, str]:
        info = p.get("info", {})
        websites = info.get("websites", [])
        website_url = websites[0].get("url", "") if websites else ""

        twitter_url = telegram_url = ""
        for s in info.get("socials", []):
            if s.get("type") == "twitter":
                twitter_url = s.get("url", "")
            elif s.get("type") == "telegram":
                telegram_url = s.get("url", "")

        base = p.get("baseToken", {})
        return (
            website_url,
            twitter_url,
            telegram_url,
            base.get("symbol", ""),
            base.get("name", ""),
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

    async def fetch_and_update_chunk(chunk_indices, chunk_df, output_filename):
        nonlocal max_attempts, initial_delay

        addresses = chunk_df["pairAddress"].tolist()
        url = (
            f"https://api.dexscreener.com/latest/dex/pairs/solana/"
            f"{','.join(addresses)}"
        )

        attempt, delay = 0, initial_delay

        while attempt < max_attempts:
            try:
                async with semaphore:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()

                            # Safely handle "pairs": null or empty
                            pairs_raw = data.get("pairs")
                            if not pairs_raw:
                                # nothing returned – write zeros and quit this chunk
                                for idx in chunk_indices:
                                    df.loc[idx, [
                                        "current_priceUsd",
                                        "current_txns_h24_buy",
                                        "current_txns_h24_sells",
                                        "current_volume_h24",
                                        "current_liquidity_usd"]] = np.nan
                                    async with lock:
                                        await asyncio.to_thread(
                                            write_row_to_csv, df.loc[idx], output_filename
                                        )
                                break

                            pairs_dict = {
                                (p.get("address") or p.get("pairAddress")): p
                                for p in pairs_raw
                                if (p.get("address") or p.get("pairAddress"))
                            }

                            for idx, old in chunk_df.iterrows():
                                pair_address = old["pairAddress"]
                                p = pairs_dict.get(pair_address)

                                # ---- update current_* ----
                                if p is None:  # API nie zwróciło rekordu dla tego pairAddress
                                    df.loc[idx, [
                                        "current_priceUsd",
                                        "current_txns_h24_buy",
                                        "current_txns_h24_sells",
                                        "current_volume_h24",
                                        "current_liquidity_usd"]] = np.nan
                                else:
                                    # ---- current_* ----
                                    df.at[idx, "current_priceUsd"]       = deep_float(p, "priceUsd")
                                    df.at[idx, "current_txns_h24_buy"]   = deep_float(p, "txns", "h24", "buys")
                                    df.at[idx, "current_txns_h24_sells"] = deep_float(p, "txns", "h24", "sells")
                                    df.at[idx, "current_volume_h24"]     = deep_float(p, "volume", "h24")
                                    df.at[idx, "current_liquidity_usd"]  = deep_float(p, "liquidity", "usd")

                                    # ---- meta fields ----
                                    website_url, twitter_url, telegram_url, sym, name = _extract_pair_meta(p)

                                    def _missing(v: str) -> bool:
                                        return pd.isna(v) or v == ""

                                    if _missing(old["website_url"])     and website_url: df.at[idx, "website_url"]     = website_url
                                    if _missing(old["twitter_url"])     and twitter_url: df.at[idx, "twitter_url"]     = twitter_url
                                    if _missing(old["telegram_url"])    and telegram_url:df.at[idx, "telegram_url"]    = telegram_url
                                    if _missing(old["baseTokenSymbol"]) and sym:         df.at[idx, "baseTokenSymbol"] = sym
                                    if _missing(old["baseTokenName"])   and name:        df.at[idx, "baseTokenName"]   = name

                                async with lock:
                                    await asyncio.to_thread(
                                        write_row_to_csv, df.loc[idx], output_filename
                                    )
                            break  # finished this chunk successfully

                        elif response.status == 429:
                            retry_after = int(response.headers.get("Retry-After", delay))
                            await asyncio.sleep(retry_after + 5)
                            delay *= 2
                            attempt += 1
                        else:
                            await asyncio.sleep(delay)
                            delay *= 2
                            attempt += 1

            except (aiohttp.ClientError, asyncio.TimeoutError):
                await asyncio.sleep(delay)
                delay *= 2
                attempt += 1

        # after max_attempts fallback
        if attempt == max_attempts:
            for idx in chunk_indices:
                df.loc[idx, [
                    "current_priceUsd",
                    "current_txns_h24_buy",
                    "current_txns_h24_sells",
                    "current_volume_h24",
                    "current_liquidity_usd"]] = np.nan
                async with lock:
                    await asyncio.to_thread(
                        write_row_to_csv, df.loc[idx], output_filename
                    )

    async with aiohttp.ClientSession(timeout=session_timeout) as session:
        tasks = [
            fetch_and_update_chunk(chunk_indices, df.loc[chunk_indices], output_filename)
            for chunk_indices in chunked(df.index.tolist(), 30)
        ]
        await asyncio.gather(*tasks)

def main():
        # --- choose local DB path ---
    local_db = None
    for cand in LOCAL_DB_CANDIDATES:
        if cand and os.path.exists(cand):
            local_db = cand
            break
    if not local_db:
        local_db = LOCAL_DB_CANDIDATES[0]
        os.makedirs(os.path.dirname(local_db), exist_ok=True)

    # --- ensure table exists (or try to bootstrap from full DB once) ---
    table_exists = False
    if os.path.exists(local_db):
        with sqlite3.connect(local_db) as con:
            q = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tokens';").fetchone()
            table_exists = q is not None

    if not table_exists:
        # First-time bootstrap: try remote full DB, otherwise fail with hint
        try:
            with requests.get(URL, stream=True, timeout=15) as r:
                r.raise_for_status()
                with open(local_db, "wb") as f:
                    shutil.copyfileobj(r.raw, f)
            table_exists = True
            print("Bootstrapped local DB from full download.")
        except requests.RequestException:
            raise FileNotFoundError(
                "No local DB with 'tokens' table and incremental source not usable for bootstrap. "
                "Expose the full DB once (URL) or place a local .db with a 'tokens' table."
            )

    # --- incremental update: fetch only rows newer than local max(pairCreatedAt) ---
    max_created_iso = None
    try:
        with sqlite3.connect(local_db) as con:
            row = con.execute("SELECT MAX(pairCreatedAt) FROM tokens;").fetchone()
            max_created_iso = (row[0] if row and row[0] else None)
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
                        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_tokens_pairAddress ON tokens(pairAddress);")
                        df_new.to_sql("tokens_new", con, if_exists="replace", index=False)
                        con.execute("INSERT OR REPLACE INTO tokens SELECT * FROM tokens_new;")
                        con.execute("DROP TABLE tokens_new;")
                    print(f"Incremental update applied: +{len(df_new)} rows.")
                else:
                    print("Incremental: no new rows.")
            else:
                print(f"Incremental fetch returned {r.status_code}; using local DB only.")
        except requests.RequestException as e:
            print(f"Incremental source not reachable ({type(e).__name__}); using local DB only.")
    else:
        print("No max(pairCreatedAt) found; skipping incremental fetch.")

    # --- read into DataFrame from local DB ---
    with sqlite3.connect(local_db) as con:
        df = pd.read_sql_query("SELECT * FROM tokens;", con)

    df.to_csv('data/tokens_fetched_db.csv', index=False)

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

    df.drop_duplicates(subset=["pairAddress"], inplace=True)
    df["pairCreatedAt"] = pd.to_datetime(df["pairCreatedAt"])
    cutoff = pd.Timestamp.now() - pd.Timedelta(hours=24)
    df = df[df["pairCreatedAt"] < cutoff]
    print(f"shape: {df.shape}")

    # --- RESUME: skip rows already written to output file ---
    output_filename = "data/tokens_raw_reclassified.csv"
    if os.path.exists(output_filename):
        try:
            completed_addresses = pd.read_csv(
                output_filename, usecols=["pairAddress"], dtype={"pairAddress": str}
            )["pairAddress"]
            completed_set = set(completed_addresses.dropna().astype(str))
            before_count = len(df)
            df = df[~df["pairAddress"].astype(str).isin(completed_set)]
            print(f"Resuming: skipped {before_count - len(df)} rows already in CSV; remaining {len(df)}.")
        except Exception as e:
            print(f"Resume warning: could not read existing CSV ({e}); processing all rows.")

    # Create header only if file does not exist yet (first run)
    if not os.path.exists(output_filename):
        with open(output_filename, "w", encoding="utf-8", newline="") as f:
            df.head(0).to_csv(f, index=False)

    print('Starting update of current_price, current_volume, and current_market_cap...')
    asyncio.run(update_worthy(df, output_filename))
    print(f"Update completed, data saved to {output_filename}")

if __name__ == "__main__":
    main()
