# TODO: Encapsulate the entire code within a class to improve organization.

import asyncio
import aiohttp
import csv
from numpy import asarray
from binance.um_futures import UMFutures
from credentials import binance_API_KEY, binance_SECRET_KEY
from talib import ADOSC, OBV, ATR, RSI, ULTOSC, TSF
import pandas as pd
from os import path
from datetime import datetime, timezone, timedelta

# Time from the token pair launch to its acceptance as a valid pair
LAUNCH_TIME = timedelta(minutes=2)
# Time during which the token pair is skipped
SKIP_TIME = timedelta(minutes=3)
# Delay for classifying the token pair after its creation
CLASSIFY_DELAY = timedelta(hours=1)
# List of time intervals for technical analysis
ITVS = ['1m', '5m', '15m', '1h', '4h', '1d']
# Global asynchronous lock used to synchronize access to the CSV file
LOCK = asyncio.Lock()


# Function to check if Binance API credentials are provided
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


check_credentials(binance_API_KEY, binance_SECRET_KEY)
CLIENT = UMFutures(binance_API_KEY, binance_SECRET_KEY)


# Fetch new tokens from geckoterminal API
async def fetch_new_tokens(session: aiohttp.ClientSession) -> dict:
    url = "https://api.geckoterminal.com/api/v2/networks/solana/new_pools"
    async with session.get(url) as response:
        response_json = await response.json()
        return response_json['data']


# Fetch details of valid pairs from dexscreener API
async def fetch_valid_pairs_details(session: aiohttp.ClientSession, address_list: list[str]) -> list[dict]:
    url = f"https://api.dexscreener.com/latest/dex/pairs/solana/{','.join(address_list)}"
    async with session.get(url) as response:
        response_json = await response.json()
        valid_pairs = []
        now = datetime.now(timezone.utc)
        for pair in response_json['pairs']:
            creation_time = datetime.fromtimestamp(pair['pairCreatedAt'] / 1000, tz=timezone.utc)
            if SKIP_TIME > (now - creation_time) > LAUNCH_TIME:
                valid_pairs.append(pair)
                print(f' valid pair address: {pair["pairAddress"]}')
        return valid_pairs


# Function to fetch OHLCV data from Binance and calculate technical analysis indicators using TA-Lib
def fetch_ta(client: UMFutures, symbol: str, itvs: list[str]) -> dict:
    ta_data = {}
    for i in itvs:
        # Fetch OHLCV data and convert it to float type
        ohlcv = asarray(client.klines(symbol=symbol,
                                      interval=i,
                                      limit=1_000)).astype(float)[:, 1:6]
        # Calculate technical analysis indicators and add them to the dictionary
        ta_data[f'{symbol}_ADOSC_{i}'] = ADOSC(ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4], fastperiod=3,
                                               slowperiod=10)[-1]
        ta_data[f'{symbol}_OBV_{i}'] = OBV(ohlcv[:, 3], ohlcv[:, 4])[-1]
        ta_data[f'{symbol}_ATR_{i}'] = ATR(ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3], timeperiod=14)[-1]
        ta_data[f'{symbol}_RSI_{i}'] = RSI(ohlcv[:, 3], timeperiod=14)[-1]
        ta_data[f'{symbol}_ULTOSC_{i}'] = ULTOSC(ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3], timeperiod1=7, timeperiod2=14,
                                                 timeperiod3=28)[-1]
        ta_data[f'{symbol}_TSF_{i}'] = TSF(ohlcv[:, 3], timeperiod=14)[-1]
    return ta_data


# Function to create a detailed dictionary for token information and add technical analysis data
async def get_details_dict(detail: dict, tas_dict: dict = None) -> dict:
    det_dict = {"chainId": detail["chainId"],
                "dexId": detail["dexId"],
                "url": detail["url"],
                "pairAddress": detail["pairAddress"],
                "baseTokenAddress": detail["baseToken"]["address"],
                "baseTokenName": detail["baseToken"]["name"],
                "baseTokenSymbol": detail["baseToken"]["symbol"],
                "quoteTokenAddress": detail["quoteToken"]["address"],
                "quoteTokenName": detail["quoteToken"]["name"],
                "quoteTokenSymbol": detail["quoteToken"]["symbol"],
                "priceNative": detail["priceNative"],
                "priceUsd": detail["priceUsd"],
                "txns_m5_buys": detail["txns"]["m5"]["buys"],
                "txns_m5_sells": detail["txns"]["m5"]["sells"],
                "txns_h1_buys": detail["txns"]["h1"]["buys"],
                "txns_h1_sells": detail["txns"]["h1"]["sells"],
                "txns_h6_buys": detail["txns"]["h6"]["buys"],
                "txns_h6_sells": detail["txns"]["h6"]["sells"],
                "txns_h24_buy": detail["txns"]["h24"]["buys"],
                "txns_h24_sells": detail["txns"]["h24"]["sells"],
                "volume_h24": detail["volume"]["h24"],
                "volume_h6": detail["volume"]["h6"],
                "volume_h1": detail["volume"]["h1"],
                "volume_m5": detail["volume"]["m5"],
                "priceChange_m5": detail["priceChange"]["m5"],
                "priceChange_h1": detail["priceChange"]["h1"],
                "priceChange_h6": detail["priceChange"]["h6"],
                "priceChange_h24": detail["priceChange"]["h24"],
                "liquidity_usd": detail["liquidity"]["usd"],
                "liquidity_base": detail["liquidity"]["base"],
                "liquidity_quote": detail["liquidity"]["quote"],
                "fdv": detail["fdv"] if '' in detail else 0.0,
                "pairCreatedAt": datetime.fromtimestamp(detail["pairCreatedAt"] / 1000).strftime('%Y-%m-%d %H:%M:%S')}
    # If tas_dict is not provided, fetch technical analysis data for SOLUSDT and BTCUSDT
    if tas_dict is None:
        tas_dict = fetch_ta(CLIENT, 'SOLUSDT', ITVS) | fetch_ta(CLIENT, 'BTCUSDT', ITVS) | {"worthy": -1}
    # Update det_dict with the technical analysis data
    det_dict.update(tas_dict)
    return det_dict


# Function to save token details to a CSV file
async def save_to_csv(session: aiohttp.ClientSession, token_details: list[dict],
                      filename: str = "data/tokens_details_ext.csv") -> None:
    # TODO: Globalise dummy_dict for speed up
    global LOCK
    # Fetch a sample dictionary of token details
    dummy_dict = (await get_details_dict(token_details[0]))
    headers = list(dummy_dict.keys())
    # Locks file until code block completed
    async with LOCK:
        file_exists = path.isfile(filename)
        with open(filename, 'a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=headers)

            if not file_exists:
                writer.writeheader()
            for detail in token_details:
                row = (await get_details_dict(detail))
                print(f' writing row {row["baseTokenName"]} {row["pairAddress"]}')
                writer.writerow(row)


# Function to classify tokens based on specific criteria
async def classify(session: aiohttp.ClientSession, price_mul: float = 1.5, fdv_mul: float = 1.2, liqU_ml: float = 1.2,
                   fdv_min: float = 1_000, liqU_min: float = 1_000) -> None:
    # TODO: Handle other than default criteria, without destroying classification model
    df = pd.read_csv('data/tokens_details_ext.csv')
    df.drop_duplicates(subset=['pairAddress'], inplace=True)
    df['pairCreatedAt'] = pd.to_datetime(df['pairCreatedAt'])
    print('Classifying...')
    for i, row in df.iterrows():
        if row['worthy'] == -1:
            creation_time = row['pairCreatedAt']
            now = datetime.now()
            if now - creation_time >= CLASSIFY_DELAY:
                url = f"https://api.dexscreener.com/latest/dex/pairs/solana/{row['pairAddress']}"
                async with session.get(url) as response:
                    cur = (await response.json())['pair']
                    # Check if the token pair meets the specified criteria
                    if (float(cur['priceUsd']) > float(row['priceUsd']) * price_mul) and (
                            cur['fdv'] > row['fdv'] * fdv_mul) and (
                            cur['liquidity']['usd'] > row['liquidity_usd'] * liqU_ml) and (
                            cur['fdv'] > fdv_min and cur['liquidity']['usd'] > liqU_min):
                        print(f'### {row["pairAddress"]} seems worthy ###')
                        print(
                            f'price_usd: {cur["priceUsd"]}/{row["priceUsd"]} fdv: {cur["fdv"]}/{row["fdv"]} liq_usd: {cur["liquidity"]["usd"]}/{row["liquidity_usd"]}')
                        df.at[i, 'worthy'] = 1
                    else:
                        print(f'{row["pairAddress"]} not worthy')
                        df.at[i, 'worthy'] = 0
            else:
                print(f'{row["pairAddress"]}, pair age: {now - creation_time}')
    df.to_csv('data/tokens_details_ext.csv', index=False)


# Main loop to continuously fetch, save, and classify tokens
async def main_loop() -> None:
    async with aiohttp.ClientSession() as session:
        while True:
            new_tokens = await fetch_new_tokens(session)
            addresses = [token['attributes']['address'] for token in new_tokens]
            if addresses:
                valid_details = await fetch_valid_pairs_details(session, addresses)
                if valid_details:
                    print(f"Saving valid pairs {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    await save_to_csv(session, valid_details)
                else:
                    print(f"No valid pairs for now {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            await classify(session)
            await asyncio.sleep(30)


if __name__ == "__main__":
    asyncio.run(main_loop())
