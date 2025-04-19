import pandas as pd
import aiohttp
import asyncio
import time
import numpy as np
import os
import math

def chunked(iterable, chunk_size):
    """Generator returning subsets (chunks) of size 'chunk_size'."""
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i : i + chunk_size]

def write_row_to_csv(row, output_filename):
    """Function to save a single row to CSV in append mode."""
    row.to_frame().T.to_csv(output_filename, mode="a", header=False, index=False)

async def update_worthy(df, output_filename):
    session_timeout = aiohttp.ClientTimeout(total=30)  # max request time
    semaphore = asyncio.Semaphore(5)  # limit concurrent requests
    lock = asyncio.Lock()  # lock for CSV writes

    max_attempts = 10
    initial_delay = 2

    async def fetch_and_update_chunk(chunk_indices, chunk_df, output_filename):
        nonlocal max_attempts, initial_delay

        addresses = chunk_df["pairAddress"].tolist()
        url = f"https://api.dexscreener.com/latest/dex/pairs/solana/{','.join(addresses)}"

        attempt = 0
        delay = initial_delay

        while attempt < max_attempts:
            try:
                async with semaphore:
                    async with session.get(url) as response:
                        if response.status == 200:
                            response_json = await response.json()

                            # Retrieve the list of pairs
                            all_pairs = response_json.get("pairs", None)
                            if all_pairs is None:
                                for idx in chunk_indices:
                                    df.at[idx, "current_priceUsd"] = 0.0
                                    df.at[idx, "current_txns_h24_buy"] = 0.0
                                    df.at[idx, "current_txns_h24_sells"] = 0.0
                                    df.at[idx, "current_volume_h24"] = 0.0
                                    df.at[idx, "current_liquidity_usd"] = 0.0
                                    async with lock:
                                        await asyncio.to_thread(write_row_to_csv, df.loc[idx], output_filename)
                                break

                            # Build a dictionary {address -> pairData}
                            pairs_dict = {}
                            for p in all_pairs:
                                pair_addr = p.get("address") or p.get("pairAddress") or ""
                                pairs_dict[pair_addr] = p

                            for idx, old in chunk_df.iterrows():
                                pair_address = old["pairAddress"]
                                pair_data = pairs_dict.get(pair_address)

                                if pair_data and "priceUsd" in pair_data and pair_data["priceUsd"] is not None:
                                    try:
                                        old_price_usd = float(old["priceUsd"])

                                        current_price = float(pair_data["priceUsd"])
                                        current_buys = float(pair_data["txns"]["h24"]["buys"])
                                        current_sells = float(pair_data["txns"]["h24"]["sells"])
                                        current_volume = float(pair_data["volume"]["h24"])
                                        current_liquidity = float(pair_data["liquidity"]["usd"])

                                        # Check only the listing price to avoid division by zero in target calculation
                                        # if old_price_usd == 0 or pd.isna(old_price_usd):
                                        #     df.at[idx, "current_priceUsd"] = 0.0
                                        #     df.at[idx, "current_txns_h24_buy"] = 0.0
                                        #     df.at[idx, "current_txns_h24_sells"] = 0.0
                                        #     df.at[idx, "current_volume_h24"] = 0.0
                                        #     df.at[idx, "current_liquidity_usd"] = 0.0
                                        #     print(f"[CHUNK] {pair_address} -> listing price invalid, set values to 0.0")
                                        # else:
                                        df.at[idx, "current_priceUsd"] = current_price
                                        df.at[idx, "current_txns_h24_buy"] = current_buys
                                        df.at[idx, "current_txns_h24_sells"] = current_sells
                                        df.at[idx, "current_volume_h24"] = current_volume
                                        df.at[idx, "current_liquidity_usd"] = current_liquidity
                                        print(f"[CHUNK] {pair_address} -> updated current_price={current_price:.8f}, current_volume={current_volume:.2f}, current_liquidity={current_liquidity:.2f}")
                                    except (ValueError, ZeroDivisionError, TypeError) as e:
                                        # df.at[idx, "current_priceUsd"] = 0.0
                                        # df.at[idx, "current_txns_h24_buy"] = 0.0
                                        # df.at[idx, "current_txns_h24_sells"] = 0.0
                                        # df.at[idx, "current_volume_h24"] = 0.0
                                        # df.at[idx, "current_liquidity_usd"] = 0.0
                                        raise ValueError(f"Error processing {pair_address}: {e}") from e
                                        # print(f"[CHUNK] Error processing {pair_address}: {e}")
                                else:
                                    df.at[idx, "current_priceUsd"] = 0.0
                                    df.at[idx, "current_txns_h24_buy"] = 0.0
                                    df.at[idx, "current_txns_h24_sells"] = 0.0
                                    df.at[idx, "current_volume_h24"] = 0.0
                                    df.at[idx, "current_liquidity_usd"] = 0.0

                                async with lock:
                                    await asyncio.to_thread(write_row_to_csv, df.loc[idx], output_filename)
                            break

                        elif response.status == 429:
                            retry_after = response.headers.get("Retry-After")
                            wait_time = int(retry_after) + 5 if retry_after else delay
                            print(f"[CHUNK] HTTP 429 Too Many Requests, waiting {wait_time} seconds")
                            await asyncio.sleep(wait_time)
                            delay *= 2
                            attempt += 1
                        else:
                            print(f"[CHUNK] HTTP error {response.status}, retrying in {delay}s")
                            await asyncio.sleep(delay)
                            delay *= 2
                            attempt += 1

            except aiohttp.ClientError as e:
                print(f"[CHUNK] Client error: {e}, retrying in {delay}s")
                await asyncio.sleep(delay)
                delay *= 2
                attempt += 1
            except asyncio.TimeoutError:
                print(f"[CHUNK] Request timeout, retrying in {delay}s")
                await asyncio.sleep(delay)
                delay *= 2
                attempt += 1
            except Exception as e:
                print(f"[CHUNK] Unexpected error: {e}")
                break

        if attempt == max_attempts:
            print(f"[CHUNK] Max attempts reached for chunk {chunk_indices}, setting values to NaN")
            for idx in chunk_indices:
                df.at[idx, "current_priceUsd"] = np.nan
                df.at[idx, "current_txns_h24_buy"] = np.nan
                df.at[idx, "current_txns_h24_sells"] = np.nan
                df.at[idx, "current_volume_h24"] = np.nan
                df.at[idx, "current_liquidity_usd"] = np.nan
                async with lock:
                    await asyncio.to_thread(write_row_to_csv, df.loc[idx], output_filename)

    async with aiohttp.ClientSession(timeout=session_timeout) as session:
        tasks = []
        all_indices = list(df.index)
        for chunk_indices in chunked(all_indices, 30):
            chunk_df = df.loc[chunk_indices]
            tasks.append(fetch_and_update_chunk(chunk_indices, chunk_df, output_filename))
        await asyncio.gather(*tasks)

def main():
    base_file = "data/tokens_raw.csv"
    if not os.path.exists(base_file):
        print("Base file tokens_raw.csv not found!")
        return
    base_df = pd.read_csv(base_file, nrows=0)
    base_columns = base_df.columns.tolist()

    # Add new columns for regression model
    additional_columns = ["current_priceUsd", "current_txns_h24_buy", "current_txns_h24_sells", "current_volume_h24", "current_liquidity_usd"]
    for col in additional_columns:
        if col not in base_columns:
            base_columns.append(col)

    csv_files = [
        "data/tokens_raw_backup.csv",
        "data/tokens_raw.csv",
        "data/tokens_raw_classified.csv",
    ]

    df_list = []
    for file in csv_files:
        if os.path.exists(file):
            temp_df = pd.read_csv(file)
            missing_cols = set(base_columns) - set(temp_df.columns)
            if missing_cols:
                for col in missing_cols:
                    temp_df[col] = np.nan
            temp_df = temp_df[base_columns]
            df_list.append(temp_df)
        else:
            print(f"File not found: {file}")

    df = pd.concat(df_list, ignore_index=True)
    df.drop_duplicates(subset=["pairAddress"], inplace=True)
    df["pairCreatedAt"] = pd.to_datetime(df["pairCreatedAt"])
    cutoff = pd.Timestamp.now() - pd.Timedelta(hours=24)
    df = df[df["pairCreatedAt"] < cutoff]
    print(f"shape: {df.shape}")

    output_filename = "data/tokens_raw_reclassified.csv"
    if not os.path.exists(output_filename):
        with open(output_filename, "w", encoding="utf-8", newline="") as f:
            df.head(0).to_csv(f, index=False)

    print('Starting update of current_price, current_volume, and current_market_cap...')
    asyncio.run(update_worthy(df, output_filename))
    print(f"Update completed, data saved to {output_filename}")

if __name__ == "__main__":
    main()
