import asyncio
import aiohttp
import csv
from binance.um_futures import UMFutures
from credentials import binance_API_KEY, binance_SECRET_KEY
import pandas as pd
from os import path
from datetime import datetime, timezone, timedelta
from common import fetch_ta


# Time from the token pair launch to its acceptance as a valid pair
LAUNCH_TIME = timedelta(minutes=0)
# Time during which the token pair is skipped
SKIP_TIME = timedelta(minutes=5)
# Delay for classifying the token pair after its creation
CLASSIFY_DELAY = timedelta(hours=24)
# List of time intervals for technical analysis
ITVS = ["1m", "5m", "15m", "1h", "4h", "1d"]
# Global asynchronous lock used to synchronize access to the CSV file
LOCK = asyncio.Lock()
DUMMY_DETAILS_DICT = None


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


# Function to create a detailed dictionary for token information and add technical analysis data
async def get_details_dict(detail: dict, tas_dict: dict = None) -> dict:
    try:
        det_dict = {
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
            "fdv": detail["fdv"] if "" in detail else 0.0,
            "pairCreatedAt": datetime.fromtimestamp(
                detail["pairCreatedAt"] / 1000
            ).strftime("%Y-%m-%d %H:%M:%S"),
        }
        # If tas_dict is not provided, fetch technical analysis data for SOLUSDT and BTCUSDT
        if tas_dict is None:
            client = UMFutures(binance_API_KEY, binance_SECRET_KEY)
            tas_dict = (
                fetch_ta(client, "SOLUSDT", ITVS)
                | fetch_ta(client, "BTCUSDT", ITVS)
                | {"worthy": -1}
            )
        # Update det_dict with the technical analysis data
        det_dict.update(tas_dict)
        return det_dict
    except KeyError as e:
        # Print details only if the missing key is not 'm5'
        if e.args[0] != "m5":
            print(f"Skipping pair with details: {detail} due to error {e}")
        else:
            print(
                f'Skipping pair with address: {detail["pairAddress"]} due to error {e}'
            )
        return False


# Function to save token details to a CSV file
async def save_to_csv(
    token_details: list[dict],
    filename: str = "data/tokens_raw.csv",
    backup_filename: str = "data/tokens_raw_backup.csv",
) -> None:
    global LOCK, DUMMY_DETAILS_DICT
    i = 0
    while not DUMMY_DETAILS_DICT:
        DUMMY_DETAILS_DICT = await get_details_dict(token_details[i])
        print(f"Failed to fetch dummy details. Trying again... {i}")
        i += 1
    headers = list(DUMMY_DETAILS_DICT.keys())
    async with LOCK:
        file_exists = path.isfile(filename)
        backup_file_exists = path.isfile(backup_filename)
        with open(filename, "a", newline="", encoding="utf-8") as file_main, open(
            backup_filename, "a", newline="", encoding="utf-8"
        ) as file_backup:
            writer_main = csv.DictWriter(file_main, fieldnames=headers)
            writer_backup = csv.DictWriter(file_backup, fieldnames=headers)

            if not file_exists:
                writer_main.writeheader()
            if not backup_file_exists:
                writer_backup.writeheader()

            client = UMFutures(binance_API_KEY, binance_SECRET_KEY)
            tas_dict = (
                fetch_ta(client, "SOLUSDT", ITVS)
                | fetch_ta(client, "BTCUSDT", ITVS)
                | {"worthy": -1}
            )
            for detail in token_details:
                try:
                    row = await get_details_dict(detail, tas_dict)
                    if row:
                        print(
                            f'Writing row {row["baseTokenName"]} {row["pairAddress"]}'
                        )
                        writer_main.writerow(row)
                        writer_backup.writerow(row)
                except KeyError as e:
                    print(
                        f'Skipping pair with address: {detail["pairAddress"]} due to error {e}'
                    )


# Function to classify tokens based on specific criteria
async def classify(
    session: aiohttp.ClientSession,
    price_mul: float = 2.0,
    fdv_mul: float = 2.0,
    liqU_ml: float = 2.0,
    fdv_min: float = 10_000,
    liqU_min: float = 10_000,
) -> None:
    # TODO: Handle other than default criteria, without destroying classification model
    async with LOCK:
        df = pd.read_csv("data/tokens_raw.csv")
        df.drop_duplicates(subset=["pairAddress"], inplace=True)
        df["pairCreatedAt"] = pd.to_datetime(df["pairCreatedAt"])
        print("Classifying...")
        for i, row in df.iterrows():
            if row["worthy"] == -1:
                time_diff = datetime.now() - row["pairCreatedAt"]
                if time_diff >= CLASSIFY_DELAY:
                    url = f"https://api.dexscreener.com/latest/dex/pairs/solana/{row['pairAddress']}"
                    # try:
                    async with session.get(url) as response:
                        cur = (await response.json())["pair"]
                        # Check if the token pair meets the specified criteria
                        if (cur is None) or (
                            not all(
                                feature in cur.keys()
                                for feature in ["priceUsd", "fdv", "liquidity"]
                            )
                        ):
                            print(
                                f'Classification of token with address {row["pairAddress"]} failed due to lack of '
                                f"key feature presence or None response. Setting worthy to 0."
                            )
                            df.at[i, "worthy"] = 0
                        elif (
                            (
                                float(cur["priceUsd"])
                                > float(row["priceUsd"]) * price_mul
                            )
                            and (cur["fdv"] > row["fdv"] * fdv_mul)
                            and (
                                cur["liquidity"]["usd"] > row["liquidity_usd"] * liqU_ml
                            )
                            and (
                                cur["fdv"] > fdv_min
                                and cur["liquidity"]["usd"] > liqU_min
                            )
                        ):
                            print(f'### {row["pairAddress"]} seems worthy ###')
                            print(
                                f'price_usd: {cur["priceUsd"]}/{row["priceUsd"]} fdv: {cur["fdv"]}/{row["fdv"]} liq_usd: {cur["liquidity"]["usd"]}/{row["liquidity_usd"]}'
                            )
                            df.at[i, "worthy"] = 1
                        else:
                            print(f'{row["pairAddress"]} not worthy')
                            df.at[i, "worthy"] = 0
                    # except TypeError as error:
                    #     print('###################################################################')
                    #     print(error)
                    #     print(f'Classification of token with address {row["pairAddress"]} failed. Setting worthy '
                    #           f'to 0.')
                    #     df.at[i, 'worthy'] = 0
                elif time_diff + timedelta(minutes=5) >= CLASSIFY_DELAY:
                    print(f'{row["pairAddress"]}, pair age: {time_diff} (~5min left)')
        print(f"Classification done. {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        df.to_csv("data/tokens_raw.csv", index=False)
        df.to_csv("data/tokens_raw_classified.csv", index=False)


# Fetch details of valid pairs from dexscreener API
async def fetch_valid_pairs_details(
    session: aiohttp.ClientSession, address_list: list[str], verbose: bool = False
) -> list[dict]:
    url = (
        f"https://api.dexscreener.com/latest/dex/pairs/solana/{','.join(address_list)}"
    )
    async with session.get(url) as response:
        response_json = await response.json()
        valid_pairs = []
        now = datetime.now(timezone.utc)
        if response_json["pairs"] is None:
            print(
                f"The URL response (pairs list) is None for list of addresses: {address_list}"
            )
            return valid_pairs
        for pair in response_json["pairs"]:
            if ("pairCreatedAt" in pair.keys()) and ("pairAddress" in pair.keys()):
                creation_time = datetime.fromtimestamp(
                    pair["pairCreatedAt"] / 1000, tz=timezone.utc
                )
                if SKIP_TIME > (now - creation_time) > LAUNCH_TIME:
                    valid_pairs.append(pair)
                    if verbose:
                        print(f' valid pair address: {pair["pairAddress"]}')
        return valid_pairs


# Fetch new tokens from geckoterminal API
async def fetch_new_tokens(session: aiohttp.ClientSession) -> dict:
    url = "https://api.geckoterminal.com/api/v2/networks/solana/new_pools"
    async with session.get(url) as response:
        response_json = await response.json()
        return response_json["data"]


# Main loop to continuously fetch, save, and classify tokens
async def main_loop() -> None:
    check_credentials(binance_API_KEY, binance_SECRET_KEY)
    async with aiohttp.ClientSession() as session:
        while True:
            print(f"Fetching new tokens...")
            new_tokens = await fetch_new_tokens(session)
            addresses = [token["attributes"]["address"] for token in new_tokens]
            if addresses:
                valid_details = await fetch_valid_pairs_details(session, addresses)
                if valid_details:
                    print(f"Saving valid pairs...")
                    await save_to_csv(valid_details)
                    print(
                        f"Valid pairs saved. {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                else:
                    print(
                        f"No valid pairs for now {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
            await classify(session)
            print(f"Sleeping for 10 seconds...")
            await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(main_loop())
