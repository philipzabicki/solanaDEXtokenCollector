# Import niezbędnych bibliotek
import asyncio
import aiohttp
import csv
import json
import pandas as pd
from os import path
from datetime import datetime, timezone, timedelta

LAUNCH_TIME = timedelta(minutes=2)
SKIP_TIME = timedelta(minutes=3)
FETCH_DELAY = timedelta(minutes=10)
LOCK = asyncio.Lock()
DELAY = timedelta(hours=1)


# Definicja funkcji asynchronicznej do pobierania nowych tokenów z GeckoTerminal
async def fetch_new_tokens(session):
    url = "https://api.geckoterminal.com/api/v2/networks/solana/new_pools"
    async with session.get(url) as response:
        response_json = await response.json()
        return response_json['data']


# Definicja funkcji asynchronicznej do pobierania szczegółów tokenów z DexScreener
async def fetch_valid_pairs_details(session, address_list):
    url = f"https://api.dexscreener.com/latest/dex/pairs/solana/{','.join(address_list)}"
    async with session.get(url) as response:
        response_json = await response.json()
        # print(json.dumps(response_json, indent=3))
        valid_pairs = []
        now = datetime.now(timezone.utc)
        for pair in response_json['pairs']:
            creation_time = datetime.fromtimestamp(pair['pairCreatedAt'] / 1000, tz=timezone.utc)
            # print(f' creation_time {creation_time} now {now}')
            if SKIP_TIME > (now - creation_time) > LAUNCH_TIME:
                valid_pairs.append(pair)
                print(f' valid pair address: {pair["pairAddress"]}')
        # print(json.dumps(valid_pairs, indent=3))
        return valid_pairs


async def classify_token_delayed(session, prev_detail_dict):
    url = f"https://api.dexscreener.com/latest/dex/pairs/solana/{prev_detail_dict['pairAddress']}"
    async with session.get(url) as response:
        response_json = await response.json()
        now = datetime.now(timezone.utc)
        creation_time = datetime.fromtimestamp(response_json['pair']['pairCreatedAt'] / 1000, tz=timezone.utc)
        while now - creation_time < FETCH_DELAY:
            print(f'{prev_detail_dict["pairAddress"]} waiting... {now - creation_time}')
            await asyncio.sleep(480)
            now = datetime.now(timezone.utc)
    async with session.get(url) as response:
        cur = (await response.json())['pair']
        cur_d = await get_details_dict(cur)
        if (cur_d['fdv'] > prev_detail_dict['fdv']) and (
                cur_d['liquidity_usd'] > prev_detail_dict['liquidity_usd']) and (
                cur_d['priceUsd'] > prev_detail_dict['priceUsd']):
            prev_detail_dict['worthy'] = 1
        else:
            prev_detail_dict['worthy'] = 0
        return prev_detail_dict


async def get_details_dict(detail):
    return {"chainId": detail["chainId"],
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
            "fdv": detail["fdv"],
            "pairCreatedAt": datetime.fromtimestamp(detail["pairCreatedAt"] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
            "worthy": -1}


# Funkcja do zapisu danych do pliku CSV
async def save_to_csv(session, token_details, filename="tokens_details.csv"):
    global LOCK
    dummy_dict = (await get_details_dict(token_details[0]))
    headers = list(dummy_dict.keys())
    # print(headers)
    async with LOCK:
        file_exists = path.isfile(filename)
        with open(filename, 'a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=headers)

            if not file_exists:
                writer.writeheader()
            for detail in token_details:
                row = (await get_details_dict(detail))
                print(f' writing row {row["baseTokenName"]} {row["pairAddress"]}')
                # classified = (await classify_token_delayed(session, prev))
                writer.writerow(row)


async def classify(session):
    df = pd.read_csv('tokens_details.csv')
    df.drop_duplicates(subset=['pairAddress'], inplace=True)
    df['pairCreatedAt'] = pd.to_datetime(df['pairCreatedAt'])
    for i, row in df.iterrows():
        if row['worthy'] == -1:
            creation_time = row['pairCreatedAt']
            now = datetime.now()
            if now - creation_time >= DELAY:
                url = f"https://api.dexscreener.com/latest/dex/pairs/solana/{row['pairAddress']}"
                async with session.get(url) as response:
                    cur = (await response.json())['pair']
                    if (cur['fdv'] > row['fdv']) and (cur['liquidity']['usd'] > row['liquidity_usd']) and (
                            float(cur['priceUsd']) > row['priceUsd']) and (
                            cur['fdv'] > 500 and cur['liquidity']['usd'] > 500):
                        print(f'fdv: {cur["fdv"]} liq_usd: {cur["liquidity"]["usd"]}')
                        print(f'{row["pairAddress"]} seems worthy')
                        df.at[i, 'worthy'] = 1
                    else:
                        print(f'{row["pairAddress"]} not worthy')
                        df.at[i, 'worthy'] = 0
            else:
                print(f'{now - creation_time} {row["pairAddress"]}')
    df.to_csv('tokens_details.csv', index=False)


async def main_loop():
    async with aiohttp.ClientSession() as session:
        while True:
            new_tokens = await fetch_new_tokens(session)

            # Pobieranie adresów tokenów
            addresses = [token['attributes']['address'] for token in new_tokens]
            # print(addresses)

            # Pobieranie szczegółów tokenów
            if addresses:
                valid_details = await fetch_valid_pairs_details(session, addresses)
                if valid_details:
                    print(f"Saving valid pairs {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    await save_to_csv(session, valid_details)
                else:
                    print(f"No valid pairs for now {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            await classify(session)
            # Odczekaj minutę przed kolejnym odpytaniem
            await asyncio.sleep(30)

asyncio.run(main_loop())
