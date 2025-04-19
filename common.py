import talib
from binance.um_futures import UMFutures
from numpy import asarray
from re import sub
from unicodedata import category

indicator_params = {
        "ADOSC": "ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume']",
        "OBV":   "ohlcv['close'], ohlcv['volume']",
        "ATR":   "ohlcv['high'], ohlcv['low'], ohlcv['close']",
        "RSI":   "ohlcv['close']",
        "ULTOSC": "ohlcv['high'], ohlcv['low'], ohlcv['close']"
    }


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


def fetch_ta_from_config(client: UMFutures, config: dict, klines_limit=100) -> dict:
    # Using TA-Lib abstract interface to dynamically call indicator functions.
    ta_data = {}
    # Iterate over each symbol and its associated indicators.
    for symbol, indicators in config.items():
        for indicator, intervals in indicators.items():
            indicator_upper = indicator.upper()
            if indicator_upper not in indicator_params:
                raise ValueError(f"Indicator '{indicator}' is not implemented.")
            param_str = indicator_params[indicator_upper]
            
            # Process each specified time interval.
            for interval in intervals:
                # Fetch OHLCV data; assume data columns order: [open, high, low, close, volume].
                data = asarray(client.klines(symbol=symbol, interval=interval, limit=klines_limit)).astype(float)[:, 1:6]
                ohlcv = {
                    'open': data[:, 0],
                    'high': data[:, 1],
                    'low': data[:, 2],
                    'close': data[:, 3],
                    'volume': data[:, 4],
                }
                # Build the evaluation string for the TA-Lib function call.
                eval_str = f"talib.{indicator_upper}({param_str})"
                # Evaluate the function call and get the last result.
                result = eval(eval_str)[-1]
                # Build a unique key and store the calculated indicator.
                ta_data[f"{symbol}_{indicator_upper}_{interval}"] = result

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