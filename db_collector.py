import asyncio
import aiohttp
import aiosqlite
from os import path, makedirs
from datetime import datetime, timezone, timedelta
import pandas as pd

# Configuration / Settings
RECLASSIFY = False  # Set to True to enable re-classification
FETCH_INTERVAL = 3  # seconds between main loop iterations
BATCH_SIZE = 15  # how many addresses to query per dexscreener request
COMPLETE_SEMAPHORE: asyncio.Semaphore = asyncio.Semaphore(100)
IMAGE_SEMAPHORE: asyncio.Semaphore = asyncio.Semaphore(30)
CSV_FILE = "data/full_tokens_raw.csv"
LAUNCH_TIME = timedelta(minutes=0)
SKIP_TIME = timedelta(minutes=10)
CLASSIFY_DELAY = timedelta(hours=24)
LOCK = asyncio.Lock()
DB_FILE = "data/tokens_raw.db"
REQUIRED_QUOTE_NAME = "Wrapped SOL"

HEADERS = [
    "chainId",
    "dexId",
    "url",
    "pairAddress",
    "baseTokenAddress",
    "baseTokenName",
    "baseTokenSymbol",
    "quoteTokenAddress",
    "quoteTokenName",
    "quoteTokenSymbol",
    "priceNative",
    "priceUsd",
    "txns_m5_buys",
    "txns_m5_sells",
    "txns_h1_buys",
    "txns_h1_sells",
    "txns_h6_buys",
    "txns_h6_sells",
    "txns_h24_buys",
    "txns_h24_sells",
    "volume_h24",
    "volume_h6",
    "volume_h1",
    "volume_m5",
    "priceChange_m5",
    "priceChange_h1",
    "priceChange_h6",
    "priceChange_h24",
    "liquidity_usd",
    "liquidity_base",
    "liquidity_quote",
    "fdv",
    "marketCap",
    "pairCreatedAt",
    "website_url",
    "twitter_url",
    "telegram_url",
]


async def initialize_db(db_file: str = DB_FILE):
    async with aiosqlite.connect(db_file) as db:
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute(
            """
        CREATE TABLE IF NOT EXISTS tokens (
            pairAddress TEXT PRIMARY KEY,
            chainId INTEGER,
            dexId TEXT,
            url TEXT,
            baseTokenAddress TEXT,
            baseTokenName TEXT,
            baseTokenSymbol TEXT,
            quoteTokenAddress TEXT,
            quoteTokenName TEXT,
            quoteTokenSymbol TEXT,
            priceNative REAL,
            priceUsd REAL,
            txns_m5_buys INTEGER,
            txns_m5_sells INTEGER,
            txns_h1_buys INTEGER,
            txns_h1_sells INTEGER,
            txns_h6_buys INTEGER,
            txns_h6_sells INTEGER,
            txns_h24_buys INTEGER,
            txns_h24_sells INTEGER,
            volume_h24 REAL,
            volume_h6 REAL,
            volume_h1 REAL,
            volume_m5 REAL,
            priceChange_m5 REAL,
            priceChange_h1 REAL,
            priceChange_h6 REAL,
            priceChange_h24 REAL,
            liquidity_usd REAL,
            liquidity_base REAL,
            liquidity_quote REAL,
            fdv REAL,
            marketCap REAL,
            pairCreatedAt TEXT,
            website_url TEXT,
            twitter_url TEXT,
            telegram_url TEXT
        );
        """
        )
        await db.commit()


async def save_to_db(
    token_details: list[dict],
    session: aiohttp.ClientSession,
    db_file: str = DB_FILE,
) -> None:
    token_details = [
        d
        for d in token_details
        if d.get("quoteToken", {}).get("name") == REQUIRED_QUOTE_NAME
    ]
    if not token_details:
        return

    rows = [flatten_pair(d) for d in token_details]

    async with LOCK:
        async with aiosqlite.connect(db_file) as db:
            await db.execute("BEGIN")
            cols = ",".join(HEADERS)
            placeholders = ",".join("?" for _ in HEADERS)
            sql = f"INSERT OR IGNORE INTO tokens ({cols}) VALUES ({placeholders})"
            await db.executemany(sql, [tuple(r[h] for h in HEADERS) for r in rows])
            await db.commit()
            print(f"✅ Zapisano {db.total_changes} nowych rekordów do bazy danych")

    # ➡️  brakujące dane i obrazy uzupełnij w tle
    for d in token_details:
        asyncio.create_task(complete_and_update(d, session))


def _safe(d: dict, *keys, default=None):
    for k in keys:
        d = d.get(k, {})
    return d or default


def flatten_pair(detail: dict) -> dict:
    info = detail.get("info", {})
    websites = info.get("websites", [])
    website_url = websites[0]["url"] if websites else None
    twitter_url = next(
        (s["url"] for s in info.get("socials", []) if s["type"] == "twitter"), None
    )
    telegram_url = next(
        (s["url"] for s in info.get("socials", []) if s["type"] == "telegram"), None
    )

    ts = detail.get("pairCreatedAt", 0) / 1000
    created = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    return {
        # strings
        "chainId": detail.get("chainId"),
        "dexId": detail.get("dexId"),
        "url": detail.get("url"),
        "pairAddress": detail.get("pairAddress"),
        "baseTokenAddress": _safe(detail, "baseToken", "address"),
        "baseTokenName": _safe(detail, "baseToken", "name"),
        "baseTokenSymbol": _safe(detail, "baseToken", "symbol"),
        "quoteTokenAddress": _safe(detail, "quoteToken", "address"),
        "quoteTokenName": _safe(detail, "quoteToken", "name"),
        "quoteTokenSymbol": _safe(detail, "quoteToken", "symbol"),
        # numbers – None ⇒ NULL w SQLite
        "priceNative": detail.get("priceNative"),
        "priceUsd": detail.get("priceUsd"),
        "txns_m5_buys": _safe(detail, "txns", "m5", "buys"),
        "txns_m5_sells": _safe(detail, "txns", "m5", "sells"),
        "txns_h1_buys": _safe(detail, "txns", "h1", "buys"),
        "txns_h1_sells": _safe(detail, "txns", "h1", "sells"),
        "txns_h6_buys": _safe(detail, "txns", "h6", "buys"),
        "txns_h6_sells": _safe(detail, "txns", "h6", "sells"),
        "txns_h24_buys": _safe(detail, "txns", "h24", "buys"),
        "txns_h24_sells": _safe(detail, "txns", "h24", "sells"),
        "volume_h24": _safe(detail, "volume", "h24"),
        "volume_h6": _safe(detail, "volume", "h6"),
        "volume_h1": _safe(detail, "volume", "h1"),
        "volume_m5": _safe(detail, "volume", "m5"),
        "priceChange_m5": _safe(detail, "priceChange", "m5"),
        "priceChange_h1": _safe(detail, "priceChange", "h1"),
        "priceChange_h6": _safe(detail, "priceChange", "h6"),
        "priceChange_h24": _safe(detail, "priceChange", "h24"),
        "liquidity_usd": _safe(detail, "liquidity", "usd"),
        "liquidity_base": _safe(detail, "liquidity", "base"),
        "liquidity_quote": _safe(detail, "liquidity", "quote"),
        "fdv": detail.get("fdv"),
        "marketCap": detail.get("marketCap"),
        "pairCreatedAt": created,
        "website_url": website_url,
        "twitter_url": twitter_url,
        "telegram_url": telegram_url,
    }


async def fetch_new_tokens(session: aiohttp.ClientSession) -> list[str]:
    print("⏳ Fetching new token addresses from GeckoTerminal...")
    url = "https://api.geckoterminal.com/api/v2/networks/solana/new_pools"
    async with session.get(url) as resp:
        data = await resp.json()
    addresses = [item["attributes"]["address"] for item in data["data"]]
    print(f"- Retrieved {len(addresses)} addresses")
    return addresses


async def fetch_valid_pairs_details(
    session: aiohttp.ClientSession, addresses: list[str], batch_size: int = BATCH_SIZE
) -> list[dict]:
    valid: list[dict] = []
    utc_now = datetime.now(timezone.utc)

    for i in range(0, len(addresses), batch_size):
        batch = addresses[i : i + batch_size]
        url = f"https://api.dexscreener.com/latest/dex/pairs/solana/{','.join(batch)}"

        async with session.get(url) as resp:
            if resp.status != 200:  # 1️⃣ sprawdzamy kod HTTP
                print(f"⚠️  HTTP {resp.status} for {url}")
                continue
            try:
                resp_json = await resp.json()
            except Exception as e:  # 2️⃣ bezpieczny JSON-decode
                print(f"⚠️  JSON error for {url}: {e}")
                continue

        # 3️⃣ .get("pairs") może zwrócić None – zamieniamy na []
        for p in resp_json.get("pairs") or []:
            if p.get("quoteToken", {}).get("name") != REQUIRED_QUOTE_NAME:
                continue

            ts = p.get("pairCreatedAt")
            if not ts:
                continue
            created = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            age = utc_now - created
            if LAUNCH_TIME < age < SKIP_TIME:
                valid.append(p)

    return valid


async def fetch_updated_detail(
    pair_address: str, session: aiohttp.ClientSession
) -> dict:
    """Fetch the latest detail for a given pairAddress from dexscreener."""
    url = f"https://api.dexscreener.com/latest/dex/pairs/solana/{pair_address}"
    async with session.get(url) as resp:
        j = await resp.json()
    return j.get("pair") or (j.get("pairs") or [None])[0] or {}


async def complete_and_update(
    detail: dict,
    session: aiohttp.ClientSession,
    db_file: str = DB_FILE,
) -> None:
    """Finish missing data and patch only fields that were empty."""
    try:
        async with COMPLETE_SEMAPHORE:
            completed = await complete_detail(detail, session)
    except Exception as e:
        print(f"❌ complete_detail failed for {detail.get('pairAddress')}: {e}")
        return

    if completed is detail:  # nothing new, skip quickly
        return

    row = flatten_pair(completed)

    try:
        async with aiosqlite.connect(db_file) as db:
            await db.execute("PRAGMA busy_timeout=5000;")  # wait if file is busy
            # fetch the existing row once
            cur = await db.execute(
                f"SELECT {','.join(HEADERS)} FROM tokens WHERE pairAddress=?",
                (row["pairAddress"],),
            )
            existing = await cur.fetchone()
            if not existing:
                return  # row disappeared meanwhile – uncommon but safe-guard

            # build a diff – update only columns that are still NULL / '' / 0
            update_cols, update_vals = [], []
            for idx, col in enumerate(HEADERS):
                old_val, new_val = existing[idx], row[col]
                if (old_val in (None, "", 0) or old_val == "") and new_val not in (
                    None,
                    "",
                    0,
                ):
                    update_cols.append(f"{col}=?")
                    update_vals.append(new_val)

            if update_cols:
                sql = f"UPDATE tokens SET {', '.join(update_cols)} WHERE pairAddress=?"
                await db.execute(sql, update_vals + [row["pairAddress"]])
                await db.commit()
                print(
                    f"🔄 Patched {len(update_cols)} missing fields for {row['pairAddress']}"
                )
    except Exception as e:
        print(f"❌ DB patch failed for {row.get('pairAddress')}: {e}")

    # start image download in the background with its own semaphore
    asyncio.create_task(_download_images_safe(completed, session))


async def complete_detail(
    detail: dict,
    session: aiohttp.ClientSession,
    max_retries: int = 42,
    retry_interval: float = 10.0,
) -> dict:
    """
    Try up to max_retries times to get m5 data; if still missing, give up and return current detail.
    """

    def has_m5(d):
        try:
            _ = d["txns"]["m5"]["buys"]
            _ = d["txns"]["m5"]["sells"]
            _ = d["priceChange"]["m5"]
            return True
        except (KeyError, TypeError):
            return False

    addr = detail.get("pairAddress", "<unknown>")
    if has_m5(detail):
        return detail

    print(f"⚠️  Incomplete data for {addr}, retrying up to {max_retries} times...")
    for attempt in range(1, max_retries + 1):
        await asyncio.sleep(retry_interval)
        detail = await fetch_updated_detail(addr, session)
        if has_m5(detail):
            print(f"   • Data complete for {addr} on attempt {attempt}")
            return detail

    print(
        f"⚠️  Still incomplete for {addr} after {max_retries} retries, proceeding anyway"
    )
    return detail


async def download_images(
    detail: dict,
    session: aiohttp.ClientSession,
    max_retries: int = 30,
    retry_interval: float = 15.0,
) -> None:
    """Download token images, retrying fetch if URLs aren't immediately available."""
    addr = detail.get("pairAddress", "unknown")
    info = detail.get("info", {})
    keys = ("imageUrl", "header", "openGraph")
    missing = [k for k in keys if not info.get(k)]
    retries = 0

    while missing and retries < max_retries:
        await asyncio.sleep(retry_interval)
        detail = await fetch_updated_detail(addr, session)
        info = detail.get("info", {})
        missing = [k for k in keys if not info.get(k)]
        retries += 1

    for key in keys:
        url = info.get(key)
        if not url:
            print(f"⚠️  No URL for {key} after {retries} retries, skipping {addr}")
            continue
        filename = f"data/imgs/{addr}_{key}.png"
        if path.isfile(filename):
            continue
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    with open(filename, "wb") as f:
                        f.write(content)
                    print(f"✅ Saved image {key} for {addr}")
                else:
                    print(f"⚠️  HTTP {resp.status} fetching {key} for {addr}")
        except Exception as e:
            print(f"❌ Error downloading {key} for {addr}: {e}")


async def _download_images_safe(detail: dict, session: aiohttp.ClientSession) -> None:
    """Download images with limited concurrency and catch every exception."""
    try:
        async with IMAGE_SEMAPHORE:
            await download_images(detail, session)
    except Exception as e:
        addr = detail.get("pairAddress", "<unknown>")
        print(f"❌ Unexpected error downloading images for {addr}: {e}")


async def classify(
    session: aiohttp.ClientSession,
    price_mul: float = 2.0,
    fdv_mul: float = 2.0,
    liq_mul: float = 2.0,
    fdv_min: float = 10_000,
    liq_min: float = 10_000,
) -> None:
    print("⏳ Starting classification step...")
    async with LOCK:
        df = pd.read_csv(CSV_FILE)
        df.drop_duplicates("pairAddress", inplace=True)
        df["pairCreatedAt"] = pd.to_datetime(df["pairCreatedAt"])
        now = datetime.now()
        to_classify = df[df["worthy"] == -1]
        print(f"   • {len(to_classify)} tokens pending classification")
        for idx, row in to_classify.iterrows():
            age = now - row["pairCreatedAt"]
            if age >= CLASSIFY_DELAY:
                url = f"https://api.dexscreener.com/latest/dex/pairs/solana/{row['pairAddress']}"
                async with session.get(url) as resp:
                    data = (await resp.json()).get("pair")
                if not data or any(
                    k not in data for k in ("priceUsd", "fdv", "liquidity")
                ):
                    df.at[idx, "worthy"] = 0
                else:
                    if (
                        float(data["priceUsd"]) > row["priceUsd"] * price_mul
                        and data["fdv"] > row["fdv"] * fdv_mul
                        and data["liquidity"]["usd"] > row["liquidity_usd"] * liq_mul
                        and data["fdv"] > fdv_min
                        and data["liquidity"]["usd"] > liq_min
                    ):
                        df.at[idx, "worthy"] = 1
                    else:
                        df.at[idx, "worthy"] = 0
        df.to_csv(CSV_FILE, index=False)
        df.to_csv("data/tokens_raw_classified.csv", index=False)
    print("✅ Classification step done")


async def main_loop() -> None:
    makedirs("data/imgs", exist_ok=True)
    seen_addresses: set[str] = set()
    await initialize_db()

    async with aiohttp.ClientSession() as session:
        while True:
            addresses = await fetch_new_tokens(session)
            new_addrs = [a for a in addresses if a not in seen_addresses]
            if new_addrs:
                valid = await fetch_valid_pairs_details(session, new_addrs)
                if valid:
                    print(f"✅ Valid addresses found: {len(valid)}")
                    await save_to_db(valid, session)
                    seen_addresses.update(
                        p["pairAddress"] for p in valid if p.get("pairAddress")
                    )
            if RECLASSIFY:
                await classify(session)
            await asyncio.sleep(FETCH_INTERVAL)


if __name__ == "__main__":
    asyncio.run(main_loop())
