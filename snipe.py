from new_pool_sniper import *
from collections import deque
import fasttext
import joblib
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
import asyncio

LAUNCH_TIME = timedelta(minutes=0)
SKIP_TIME = timedelta(minutes=2)
PRED_ENUMS = {0: 'AVOID', 1: "BUY"}


async def get_input(df: pd.DataFrame, scaler, names_model, symbols_model) -> pd.DataFrame:
    # Extract word vectors for token names and symbols using the FastText models
    names_vectors = df['baseTokenName'].apply(lambda x: pd.Series(names_model.get_word_vector(x)))
    names_vectors.columns = [f'nameVectorDim{i}' for i in range(24)]
    symbols_vectors = df['baseTokenSymbol'].apply(lambda x: pd.Series(symbols_model.get_word_vector(x)))
    symbols_vectors.columns = [f'symbolVectorDim{i}' for i in range(12)]

    # Remove original name and symbol columns
    df.drop(columns=['baseTokenName', 'baseTokenSymbol'], inplace=True)

    # Concatenate the original DataFrame with the new vector columns
    ret = pd.concat([df, names_vectors, symbols_vectors], axis=1)

    # Transform the DataFrame using the provided scaler
    return scaler.transform(ret)


async def get_features_df(detail: dict, tas_dict: dict = None) -> pd.DataFrame:
    # Construct a dictionary with various token details and transaction statistics
    det_dict = {
        "baseTokenName": detail["baseToken"]["name"],
        "baseTokenSymbol": detail["baseToken"]["symbol"],
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
        "fdv": detail["fdv"] if 'fdv' in detail else 0.0,  # Check if 'fdv' exists in detail
    }

    # Calculate liquidity to FDV ratio, set to 0 if conditions are not met
    liq = {"liq_fdv_ratio": detail["liquidity_usd"] / detail["fdv"] if "liquidity_usd" in detail and detail[
        "fdv"] > 0 else 0.0}

    # Fetch technical analysis data if not provided
    if tas_dict is None:
        tas_dict = fetch_ta(CLIENT, 'SOLUSDT', ITVS) | fetch_ta(CLIENT, 'BTCUSDT', ITVS) | liq

    # Update the details dictionary with technical analysis and liquidity data
    det_dict.update(tas_dict | liq)

    # Convert the dictionary to a DataFrame
    return pd.DataFrame([det_dict])


async def main():
    # Load token names FastText model
    model_n = fasttext.load_model('models/token_names_model.bin')
    # Load token symbols FastText model
    model_s = fasttext.load_model('models/token_symbols_model.bin')
    # Load scaler model
    scaler = joblib.load('models/scaler_model.pkl')
    # Load predictive model
    rf_model = joblib.load('models/rf_model.joblib')

    seen_addresses = deque(maxlen=200)  # Track seen addresses with a maximum length

    async with aiohttp.ClientSession() as session:
        tas_dict = fetch_ta(CLIENT, 'SOLUSDT', ITVS) | fetch_ta(CLIENT, 'BTCUSDT', ITVS)

        while True:
            new_tokens = await fetch_new_tokens(session)

            # Extract unique token addresses that have not been seen before
            addresses = list(set(token['attributes']['address'] for token in new_tokens) - set(seen_addresses))

            if addresses:
                valid_details = await fetch_valid_pairs_details(session, addresses)
                if valid_details:
                    for token in valid_details:
                        seen_addresses.append(token['pairAddress'])  # Append the seen address
                        features_df = await get_features_df(token, tas_dict)
                        input_data = await get_input(features_df, scaler, model_n, model_s)
                        pred = rf_model.predict(input_data)

                        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', end=' ')
                        print(
                            f'Prediction: {PRED_ENUMS[pred[0]]} Name: {token["baseToken"]["name"]} Symbol: {token["baseToken"]["symbol"]}',
                            end=' ')
                        print(f'url: {token["url"]}')
                else:
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} No valid pairs for now...")

            # Wait for a minute before fetching new tokens again
            await asyncio.sleep(60)  # Note: Increased sleep time to 60 seconds for practical purposes
            tas_dict = fetch_ta(CLIENT, 'SOLUSDT', ITVS) | fetch_ta(CLIENT, 'BTCUSDT', ITVS)


if __name__ == "__main__":
    asyncio.run(main())
