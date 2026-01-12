import talib
import aiohttp
import numpy as np
import pandas as pd
from binance.um_futures import UMFutures
from numpy import asarray
from re import sub
from unicodedata import category

FLOOR_RULES = {
    "1m": "min",
    "5m": "5min",
    "15m": "15min",
    "1h": "h",
    "4h": "4h",
    "1d": "D",
}

INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

SHIFTS = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}

indicator_params = {
    "ADX": ["high", "low", "close"],
    "ADOSC": ["high", "low", "close", "volume"],
    "APO": ["close"],
    "AROONOSC": ["high", "low"],
    "BOP": ["open", "high", "low", "close"],
    "CCI": ["high", "low", "close"],
    "CMO": ["close"],
    "MFI": ["high", "low", "close", "volume"],
    "MOM": ["close"],
    "NATR": ["high", "low", "close"],
    "ROCP": ["close"],
    "RSI": ["close"],  # ← dodany wskaźnik
    "STOCHF": ["high", "low", "close"],
    "STOCHRSI": ["close"],
    "ULTOSC": ["high", "low", "close"],
    "WILLR": ["high", "low", "close"],
}


def check_credentials(api_key: str, secret_key: str) -> None:
    if not api_key or not secret_key:
        raise ValueError(
            "Binance API key and secret key must be provided in credentials.py.\n"
            "It's used to collect real-time market data from exchange for features extraction.\n"
            "Please open the credentials.py file and ensure it contains the following lines with your API credentials:\n\n"
            "binance_API_KEY = 'your_api_key_here'\n"
            "binance_SECRET_KEY = 'your_secret_key_here'\n\n"
            "How to get API: https://www.binance.com/pl/binance-api"
        )


async def fetch_new_tokens(session):
    url = "https://api.geckoterminal.com/api/v2/networks/solana/new_pools"
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
        try:
            payload = await response.json()
        except Exception:
            text = await response.text()
            print(
                f"[fetch_new_tokens] NON-JSON status={response.status} body={text[:300]}"
            )
            return []

        data = payload.get("data")
        if not isinstance(data, list):
            print(
                f"[fetch_new_tokens] NO 'data' status={response.status} keys={list(payload)[:20]} payload={payload}"
            )
            return []

        return data


def clean_string(s):
    if isinstance(s, str):
        # Usuń niewidoczne znaki kontrolne
        s = "".join(c for c in s if category(c)[0] != "C")
        # Zastąp nadmiarowe białe znaki pojedynczą spacją
        s = sub(r"\s+", " ", s)
        # Usuń białe znaki z początku i końca
        return s.strip()
    else:
        return s


def fetch_ta_from_config(client, config, klines_limit=100, ref_ms=None, ta_cache=None):
    # ref_ms: timestamp odniesienia w ms (UTC) -> pairCreatedAt
    if ref_ms is None:
        # fallback (jakbyś chciał kiedyś liczyć "dla teraz")
        ref_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)

    ta_data = {}

    for symbol, indicators in config.items():
        intervals_map = {}
        for ind, ivals in indicators.items():
            up = ind.upper()
            if up not in indicator_params:
                raise ValueError(f"Indicator '{ind}' is not implemented.")
            for interval in ivals:
                intervals_map.setdefault(interval, []).append(up)

        for interval, inds in intervals_map.items():
            iv_ms = INTERVAL_MS[interval]
            close_ms = (ref_ms // iv_ms) * iv_ms  # np. 12:34:56 -> 12:34:00
            end_ms = close_ms - 1  # gwarantuje brak świecy z open==close_ms

            cache_key = None
            if ta_cache is not None:
                cache_key = (symbol, interval, close_ms, klines_limit)
                cached = ta_cache.get(cache_key)
                if cached is not None:
                    ta_data.update(cached)
                    continue

            raw = np.asarray(
                client.klines(
                    symbol=symbol, interval=interval, endTime=end_ms, limit=klines_limit
                ),
                dtype=float,
            )

            o, h, l, c, v = raw[:, 1], raw[:, 2], raw[:, 3], raw[:, 4], raw[:, 5]
            ohlcv = {"open": o, "high": h, "low": l, "close": c, "volume": v}

            one = {}
            for up in inds:
                fn = getattr(talib, up)
                args = [ohlcv[p] for p in indicator_params[up]]
                res = fn(*args)
                arr = res[-1] if isinstance(res, tuple) else res
                one[f"{symbol}{interval}_{up}"] = float(arr[-1])

            if ta_cache is not None and cache_key is not None:
                ta_cache[cache_key] = one

            ta_data.update(one)

    return ta_data


def fetch_ta(client: UMFutures, symbol: str, itvs: list[str], klines_limit=100) -> dict:
    ta_data = {}
    for i in itvs:
        # Fetch OHLCV data and convert it to float type
        ohlcv = asarray(
            client.klines(symbol=symbol, interval=i, limit=klines_limit)
        ).astype(float)[:, 1:6]
        # Calculate technical analysis indicators and add them to the dictionary
        ta_data[f"{symbol}_ADOSC_{i}"] = talib.ADOSC(
            ohlcv[:, 1],
            ohlcv[:, 2],
            ohlcv[:, 3],
            ohlcv[:, 4],
            fastperiod=3,
            slowperiod=10,
        )[-1]
        ta_data[f"{symbol}_OBV_{i}"] = talib.OBV(ohlcv[:, 3], ohlcv[:, 4])[-1]
        ta_data[f"{symbol}_ATR_{i}"] = talib.ATR(
            ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3], timeperiod=14
        )[-1]
        ta_data[f"{symbol}_RSI_{i}"] = talib.RSI(ohlcv[:, 3], timeperiod=14)[-1]
        ta_data[f"{symbol}_ULTOSC_{i}"] = talib.ULTOSC(
            ohlcv[:, 1],
            ohlcv[:, 2],
            ohlcv[:, 3],
            timeperiod1=7,
            timeperiod2=14,
            timeperiod3=28,
        )[-1]
        ta_data[f"{symbol}_TSF_{i}"] = talib.TSF(ohlcv[:, 3], timeperiod=14)[-1]
    return ta_data
