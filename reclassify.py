import pandas as pd
import aiohttp
import asyncio
import time
import numpy as np
import aiofiles
import os

def write_row_to_csv(row, filename):
    with open(filename, 'a', encoding='utf-8', newline='') as f:
        row.to_frame().T.to_csv(f, header=False, index=False)

async def update_worthy(df, output_filename):
    session_timeout = aiohttp.ClientTimeout(total=30)  # Set a timeout for requests
    semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
    lock = asyncio.Lock()  # Lock for synchronizing file writes

    async with aiohttp.ClientSession(timeout=session_timeout) as session:
        tasks = []

        async def fetch_and_update(idx, old, output_filename):
            async with semaphore:
                attempt = 0
                max_attempts = 10
                delay = 2  # Initial delay in seconds
                pairAddress = old['pairAddress']

                while attempt < max_attempts:
                    try:
                        url = f"https://api.dexscreener.com/latest/dex/pairs/solana/{pairAddress}"
                        async with session.get(url) as response:
                            if response.status == 200:
                                response_json = await response.json()
                                cur = response_json.get('pair', None)
                                if cur and 'priceUsd' in cur and cur['priceUsd'] is not None:
                                    try:
                                        new_priceUsd = float(cur['priceUsd'])
                                        old_priceUsd_value = float(old['priceUsd'])
                                        if (old_priceUsd_value == 0) or pd.isna(old_priceUsd_value) or (float(cur["liquidity"]["usd"]) < old['liquidity_usd']*.01) or (float(cur["volume"]["h24"]) < old['volume_h24']*.01) or (float(cur["txns"]["h24"]["buys"]) < old['txns_h24_buy']*.01):
                                            print(f'Old priceUsd is zero or liq/volume/txns is lower for {pairAddress}, setting worthy to 0.0')
                                            df.at[idx, 'worthy'] = 0.0
                                        else:
                                            ratio = new_priceUsd / old_priceUsd_value
                                            df.at[idx, 'worthy'] = ratio
                                            print(f'Updated "worthy" for pairAddress {pairAddress} to {ratio}')
                                    except (ValueError, ZeroDivisionError, TypeError) as e:
                                        print(f'Error processing pairAddress {pairAddress}: {e}, setting worthy to 0.0')
                                        df.at[idx, 'worthy'] = 0.0
                                else:
                                    # print(f'No data for pairAddress {pairAddress}, setting worthy to 0.0')
                                    df.at[idx, 'worthy'] = 0.0

                                # Zapisz zaktualizowany wiersz do pliku CSV
                                async with lock:
                                    await asyncio.to_thread(write_row_to_csv, df.loc[idx], output_filename)

                                break  # Successful request; exit the loop
                            elif response.status == 429:
                                # Handle rate limiting
                                retry_after = response.headers.get("Retry-After")
                                if retry_after:
                                    wait_time = int(retry_after)+5
                                else:
                                    wait_time = delay
                                print(f'HTTP 429 Too Many Requests for pairAddress {pairAddress}, waiting {wait_time} seconds')
                                await asyncio.sleep(wait_time)
                                delay *= 2  # Exponential backoff
                                attempt += 1
                            else:
                                print(f'HTTP error {response.status} for pairAddress {pairAddress}, retrying after {delay} seconds')
                                await asyncio.sleep(delay)
                                delay *= 2
                                attempt += 1
                    except aiohttp.ClientError as e:
                        print(f'Client error for pairAddress {pairAddress}: {e}, retrying after {delay} seconds')
                        await asyncio.sleep(delay)
                        delay *= 2
                        attempt += 1
                    except asyncio.TimeoutError:
                        print(f'Timeout error for pairAddress {pairAddress}, retrying after {delay} seconds')
                        await asyncio.sleep(delay)
                        delay *= 2
                        attempt += 1
                    except Exception as e:
                        print(f'Unexpected error for pairAddress {pairAddress}: {e}')
                        break

                if attempt == max_attempts:
                    print(f'Max retries reached for pairAddress {pairAddress}, marking worthy as NaN')
                    df.at[idx, 'worthy'] = np.nan
                    # Zapisz zaktualizowany wiersz do pliku CSV nawet w przypadku niepowodzenia
                    async with lock:
                        await asyncio.to_thread(write_row_to_csv, df.loc[idx], output_filename)

        # Iterate over the DataFrame and create tasks
        for idx, row in df.iterrows():
            tasks.append(fetch_and_update(idx, row, output_filename))

        await asyncio.gather(*tasks)

def main():
    # Load the existing CSV file
    df = pd.read_csv('data/tokens_raw_classified.csv')
    df.drop_duplicates(subset=['pairAddress'], inplace=True)
    df['pairCreatedAt'] = pd.to_datetime(df['pairCreatedAt'])

    output_filename = 'data/tokens_raw_classified_v2.csv'

    # Jeśli plik nie istnieje, zapisz nagłówek
    if not os.path.exists(output_filename):
        with open(output_filename, 'w', encoding='utf-8', newline='') as f:
            df.head(0).to_csv(f, index=False)

    # Update the 'worthy' column
    print('Starting the update of the "worthy" column...')
    asyncio.run(update_worthy(df, output_filename))

    print(f'Update completed and data saved to {output_filename}')

if __name__ == "__main__":
    main()
