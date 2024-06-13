from numpy import asarray
from binance.um_futures import UMFutures
from credentials import binance_API_KEY, binance_SECRET_KEY
from talib import ADOSC, OBV, ATR, RSI, ULTOSC, TSF


def fetch_ta(client, symbol, itvs):
    ta_data = {}
    for i in itvs:
        ohlcv = asarray(client.klines(symbol=symbol,
                                      interval=i,
                                      limit=1_000)).astype(float)[:, 1:6]
        ta_data[f'{symbol}_ADOSC_{i}'] = ADOSC(ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4], fastperiod=3,
                                      slowperiod=10)[-1]
        ta_data[f'{symbol}_OBV_{i}'] = OBV(ohlcv[:, 3], ohlcv[:, 4])[-1]
        ta_data[f'{symbol}_ATR_{i}'] = ATR(ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3], timeperiod=14)[-1]
        ta_data[f'{symbol}_RSI_{i}'] = RSI(ohlcv[:, 3], timeperiod=14)[-1]
        ta_data[f'{symbol}_ULTOSC_{i}'] = ULTOSC(ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3], timeperiod1=7, timeperiod2=14,
                                        timeperiod3=28)[-1]
        ta_data[f'{symbol}_TSF_{i}'] = TSF(ohlcv[:, 3], timeperiod=14)[-1]
    return ta_data


if __name__ == "__main__":
    client = UMFutures(binance_API_KEY, binance_SECRET_KEY)
    itvs = ['1m', '5m', '15m', '1h', '4h', '1d']
    d = fetch_ta(client,'SOLUSDT', itvs)
    print(d)
