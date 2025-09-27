import talib
import aiohttp
import numpy as np
from binance.um_futures import UMFutures
from typing import Dict, List
from numpy import asarray
from re import sub
from unicodedata import category

indicator_params = {
    "ADX":       ["high", "low", "close"],
    "ADOSC":     ["high", "low", "close", "volume"],
    "APO":       ["close"],
    "AROONOSC":  ["high", "low"],
    "BOP":       ["open", "high", "low", "close"],
    "CCI":       ["high", "low", "close"],
    "CMO":       ["close"],
    "MFI":       ["high", "low", "close", "volume"],
    "MOM":       ["close"],
    "NATR":      ["high", "low", "close"],
    "ROCP":      ["close"],
    "RSI":       ["close"],          # ← dodany wskaźnik
    "STOCHF":    ["high", "low", "close"],
    "STOCHRSI":  ["close"],
    "ULTOSC":    ["high", "low", "close"],
    "WILLR":     ["high", "low", "close"],
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


async def fetch_new_tokens(session: aiohttp.ClientSession) -> dict:
    url = "https://api.geckoterminal.com/api/v2/networks/solana/new_pools"
    async with session.get(url) as response:
        response_json = await response.json()
        return response_json["data"]
    

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


def fetch_ta_from_config(client, config, klines_limit=100):
    ta_data = {}
    for symbol, indicators in config.items():
        # odwracamy mapping: dla każdego interwału lista wskaźników
        intervals_map = {}
        for ind, ivals in indicators.items():
            up = ind.upper()
            if up not in indicator_params:
                raise ValueError(f"Indicator '{ind}' is not implemented.")
            for interval in ivals:
                intervals_map.setdefault(interval, []).append(up)
        for interval, inds in intervals_map.items():
            raw = np.asarray(
                client.klines(symbol=symbol, interval=interval, limit=klines_limit),
                dtype=float
            )
            raw = raw[:-1]  # odrzucamy ostatnią niezamkniętą świecę
            o, h, l, c, v = raw[:,1], raw[:,2], raw[:,3], raw[:,4], raw[:,5]
            ohlcv = {"open": o, "high": h, "low": l, "close": c, "volume": v}
            for up in inds:
                fn = getattr(talib, up)
                args = [ohlcv[p] for p in indicator_params[up]]
                res = fn(*args)
                arr = res[-1] if isinstance(res, tuple) else res
                ta_data[f"{symbol}{interval}_{up}"] = float(arr[-1])
    return ta_data


def fetch_ta(client: UMFutures, symbol: str, itvs: list[str], klines_limit=100) -> dict:
    ta_data = {}
    for i in itvs:
        # Fetch OHLCV data and convert it to float type
        ohlcv = asarray(client.klines(symbol=symbol, interval=i, limit=klines_limit)).astype(
            float
        )[:, 1:6]
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