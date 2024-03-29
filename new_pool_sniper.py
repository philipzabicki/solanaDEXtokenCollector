# Import niezbędnych bibliotek
import asyncio
import aiohttp
import csv
import json
from os import path
from datetime import datetime, timezone, timedelta

LAUNCH_TIME = timedelta(minutes=2)
SKIP_TIME = timedelta(minutes=3)
LOCK = asyncio.Lock()


# Definicja funkcji asynchronicznej do pobierania nowych tokenów z GeckoTerminal
async def fetch_new_tokens(session):
    url = "https://api.geckoterminal.com/api/v2/networks/solana/new_pools"
    async with session.get(url) as response:
        response_json = await response.json()
        return response_json['data']


# Definicja funkcji asynchronicznej do pobierania szczegółów tokenów z DexScreener
async def fetch_token_details(session, address_list):
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


# Funkcja do zapisu danych do pliku CSV
async def save_to_csv(token_details, filename="tokens_details.csv"):
    global LOCK
    # Definiowanie nagłówków dla pliku CSV
    headers = [
        'chainId',
        'dexId',
        'url',
        'pairAddress',
        'baseTokenAddress',
        'baseTokenName',
        'baseTokenSymbol',
        'quoteTokenAddress',
        'quoteTokenName',
        'quoteTokenSymbol',
        'priceNative',
        'priceUsd',
        'txns_m5_buys',
        'txns_m5_sells',
        'txns_h1_buys',
        'txns_h1_sells',
        'txns_h6_buys',
        'txns_h6_sells',
        'txns_h24_buy',
        'txns_h24_sells',
        'volume_h24',
        'volume_h6',
        'volume_h1',
        'volume_m5',
        'priceChange_m5',
        'priceChange_h1',
        'priceChange_h6',
        'priceChange_h24',
        'liquidity_usd',
        'liquidity_base',
        'liquidity_quote',
        'fdv',
        'pairCreatedAt'
    ]
    async with LOCK:
        file_exists = path.isfile(filename)

        with open(filename, 'a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=headers)

            if not file_exists:
                writer.writeheader()

            # Przechodzenie przez każdy detal tokenu i zapisywanie go do pliku CSV
            for detail in token_details:
                row = {
                    "chainId": detail["chainId"],
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
                    "pairCreatedAt": datetime.fromtimestamp(detail["pairCreatedAt"] / 1000).strftime('%Y-%m-%d %H:%M:%S')
                }
                writer.writerow(row)


async def main_loop():
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                # Pobranie nowych tokenów
                new_tokens = await fetch_new_tokens(session)

                # Pobieranie adresów tokenów
                addresses = [token['attributes']['address'] for token in new_tokens]

                # Pobieranie szczegółów tokenów
                if addresses:
                    token_details = await fetch_token_details(session, addresses)
                    # Zapis do CSV
                    await save_to_csv(token_details)
                    # await asyncio.sleep(0.1)
                print(f"Zaktualizowano dane o tokenach: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception as e:
                print(f"Wystąpił błąd: {e}")

            # Odczekaj minutę przed kolejnym odpytaniem
            await asyncio.sleep(5)


asyncio.run(main_loop())
