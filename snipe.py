import joblib
import aiohttp
import pandas as pd
import asyncio
import webbrowser
from fasttext import load_model
from new_pool_sniper import ITVS, fetch_valid_pairs_details, fetch_new_tokens, check_credentials, fetch_ta
from credentials import binance_API_KEY, binance_SECRET_KEY
from binance.um_futures import UMFutures
from collections import deque
from datetime import datetime, timedelta

LAUNCH_TIME = timedelta(minutes=0)
SKIP_TIME = timedelta(minutes=2)
PRED_ENUMS = {0: 'AVOID', 1: "BUY"}


async def get_input(df: pd.DataFrame, scaler, names_model, symbols_model) -> pd.DataFrame:
    # Extract word vectors for token names and symbols using the FastText models
    names_vectors = df['baseTokenName'].apply(lambda x: pd.Series(names_model.get_word_vector(x)))
    names_vectors.columns = [f'nameVectorDim{i}' for i in range(30)]
    symbols_vectors = df['baseTokenSymbol'].apply(lambda x: pd.Series(symbols_model.get_word_vector(x)))
    symbols_vectors.columns = [f'symbolVectorDim{i}' for i in range(15)]

    # Remove original name and symbol columns
    df.drop(columns=['baseTokenName', 'baseTokenSymbol'], inplace=True)

    # Concatenate the original DataFrame with the new vector columns
    ret = pd.concat([df, names_vectors, symbols_vectors], axis=1)

    # Transform the DataFrame using the provided scaler
    return pd.DataFrame(scaler.transform(ret), columns=ret.columns).drop(columns='worthy')


async def get_features_df(detail: dict, tas_dict: dict = None) -> pd.DataFrame:
    # Construct a dictionary with various token details and transaction statistics
    try:
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
    except KeyError:
        print(f'Creation of features for pair address {detail["pairAddress"]} failed.')
        print(KeyError)
        return pd.DataFrame()

    # Calculate liquidity to FDV ratio, set to 0 if conditions are not met
    liq = {"worthy": -1,
           "liq_fdv_ratio": detail["liquidity_usd"] / detail["fdv"] if "liquidity_usd" in detail and detail["fdv"] > 0 else 0.0}

    # Fetch technical analysis data if not provided
    if tas_dict is None:
        client = UMFutures(binance_API_KEY, binance_SECRET_KEY)
        tas_dict = fetch_ta(client, 'SOLUSDT', ITVS) | fetch_ta(client, 'BTCUSDT', ITVS) | liq

    # Update the details dictionary with technical analysis and liquidity data
    det_dict.update(tas_dict | liq)

    # print(det_dict.keys())
    # Convert the dictionary to a DataFrame
    return pd.DataFrame([det_dict])


async def main():
    check_credentials(binance_API_KEY, binance_SECRET_KEY)
    # Load token names FastText model
    model_n = load_model('models/names2vec_model.bin')
    # Load token symbols FastText model
    model_s = load_model('models/symbols2vec_model.bin')
    # Load scaler model
    scaler = joblib.load('models/std_scaler.pkl')
    # Load predictive models
    rf_model = joblib.load('models/default_rf_model.joblib')
    svm_model = joblib.load('models/default_svm_model.joblib')
    lr_model = joblib.load('models/default_lr_model.joblib')
    gb_model = joblib.load('models/default_gb_model.joblib')

    seen_addresses = deque(maxlen=500)  # Track seen addresses with a maximum length

    async with aiohttp.ClientSession() as session:
        client = UMFutures(binance_API_KEY, binance_SECRET_KEY)
        tas_dict = fetch_ta(client, 'SOLUSDT', ITVS) | fetch_ta(client, 'BTCUSDT', ITVS)

        while True:
            new_tokens = await fetch_new_tokens(session)

            # Extract unique token addresses that have not been seen before
            addresses = list(set(token['attributes']['address'] for token in new_tokens) - set(seen_addresses))

            if addresses:
                valid_details = await fetch_valid_pairs_details(session, addresses, verbose=False)
                if valid_details:
                    for token in valid_details:
                        seen_addresses.append(token['pairAddress'])  # Append the seen address
                        features_df = await get_features_df(token, tas_dict)
                        if not features_df.empty:
                            input_data = await get_input(features_df, scaler, model_n, model_s)
                            # input_data = np.delete(input_data, -47, axis=1)
                            # print(type(input_data))
                            # print(input_data[:,-47])
                            rf_pred = rf_model.predict(input_data)
                            svm_pred = svm_model.predict(input_data)
                            lr_pred = lr_model.predict(input_data)
                            gb_pred = gb_model.predict(input_data)
                            sum_preds = int(rf_pred[0]+svm_pred[0]+lr_pred[0]+gb_pred[0])
                            print('##################################################')
                            print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}')
                            print(f'PREDICTIONS')
                            print(f' summary: {sum_preds}/4')
                            print(f' models RF/SVM/GB/LR: {PRED_ENUMS[rf_pred[0]]}/{PRED_ENUMS[svm_pred[0]]}/{PRED_ENUMS[gb_pred[0]]}/{PRED_ENUMS[lr_pred[0]]}')
                            print(f'Name: {token["baseToken"]["name"]} Symbol: {token["baseToken"]["symbol"]}')
                            print(f'url: {token["url"]}')
                            print('##################################################')
                            if sum_preds > 2:
                                webbrowser.open(token["url"])
                else:
                    pass
                    # print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} No valid pairs for now...")

            # Wait for a minute before fetching new tokens again
            await asyncio.sleep(60)  # Note: Increased sleep time to 60 seconds for practical purposes
            tas_dict = fetch_ta(client, 'SOLUSDT', ITVS) | fetch_ta(client, 'BTCUSDT', ITVS)


if __name__ == "__main__":
    asyncio.run(main())
