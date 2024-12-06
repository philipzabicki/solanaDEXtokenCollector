import joblib
import aiohttp
import pandas as pd
import asyncio
import webbrowser
from numpy import mean
from beepy import beep
from fasttext import load_model
from new_pool_sniper import ITVS, fetch_valid_pairs_details, fetch_new_tokens, check_credentials, fetch_ta
from credentials import *
from binance.um_futures import UMFutures
from collections import deque
from datetime import datetime, timedelta
import re
import unicodedata


LAUNCH_TIME = timedelta(minutes=0)
SKIP_TIME = timedelta(minutes=3)
PRED_ENUMS = {0: 'AVOID', 1: "BUY"}
CHAT_IDs = ['5011277677', '7228159263']


def clean_string(s):
    if isinstance(s, str):
        # Usuń niewidoczne znaki kontrolne
        s = ''.join(c for c in s if unicodedata.category(c)[0] != 'C')
        # Zastąp nadmiarowe białe znaki pojedynczą spacją
        s = re.sub(r'\s+', ' ', s)
        # Usuń białe znaki z początku i końca
        return s.strip()
    else:
        return s

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
            "baseTokenName": clean_string(detail["baseToken"]["name"]),
            "baseTokenSymbol": clean_string(detail["baseToken"]["symbol"]),
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


async def send_telegram_message(session, message: str, chat_ids: list):
    url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
    for chat_id in chat_ids:
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'Markdown'  # Opcjonalnie, możesz użyć innego formatu
        }
        async with session.post(url, data=payload) as response:
            if response.status != 200:
                print(f'Failed to send message to {chat_id}: {response.status}')
                resp_text = await response.text()
                print(resp_text)


async def main():
    check_credentials(binance_API_KEY, binance_SECRET_KEY)
    # Load models
    model_n = load_model('models/names2vec_model.bin')
    model_s = load_model('models/symbols2vec_model.bin')
    scaler = joblib.load('models/std_scaler.pkl')
    rf_model = joblib.load('models/default_rf_model.joblib')
    # svm_model = joblib.load('models/default_svm_model.joblib')
    lr_model = joblib.load('models/default_lr_model.joblib')
    gb_model = joblib.load('models/default_xgb_model.joblib')

    seen_addresses = deque(maxlen=500)

    async with aiohttp.ClientSession() as session:
        client = UMFutures(binance_API_KEY, binance_SECRET_KEY)
        tas_dict = fetch_ta(client, 'SOLUSDT', ITVS) | fetch_ta(client, 'BTCUSDT', ITVS)

        while True:
            new_tokens = await fetch_new_tokens(session)
            addresses = list(set(token['attributes']['address'] for token in new_tokens) - set(seen_addresses))

            if addresses:
                valid_details = await fetch_valid_pairs_details(session, addresses, verbose=False)
                if valid_details:
                    for token in valid_details:
                        seen_addresses.append(token['pairAddress'])
                        features_df = await get_features_df(token, tas_dict)
                        if not features_df.empty:
                            input_data = await get_input(features_df, scaler, model_n, model_s)

                            # Uzyskanie prawdopodobieństw zamiast predykcji binarnych
                            rf_proba = rf_model.predict_proba(input_data)[0][1]
                            # svm_proba = svm_model.predict_proba(input_data)[0][1]
                            lr_proba = lr_model.predict_proba(input_data)[0][1]
                            gb_proba = gb_model.predict_proba(input_data)[0][1]

                            # Konwersja prawdopodobieństw na wartości binarne dla logiki sumowania
                            # rf_pred = int(rf_proba > 0.5)
                            # svm_pred = int(svm_proba > 0.5)
                            # lr_pred = int(lr_proba > 0.5)
                            # gb_pred = int(gb_proba > 0.5)

                            # sum_preds = rf_proba + lr_proba + gb_proba
                            mean_preds = mean([rf_proba, lr_proba, gb_proba], axis=0)

                            if mean_preds > 0.5:
                                for _ in range(2): beep(sound='coin')
                                print('##################################################')
                                print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}')
                                print(f'PREDICTIONS')
                                print(f'Prob mean: {mean_preds * 100:.2f}%')
                                print(
                                    f"RF: {rf_proba * 100:.2f}%\n"
                                    f"xGB: {gb_proba * 100:.2f}%\n"
                                    f"LR: {lr_proba * 100:.2f}%\n"
                                )
                                print(f'Name: {token["baseToken"]["name"]} Symbol: {token["baseToken"]["symbol"]}')
                                print(f'url: {token["url"]}')
                                print('##################################################')

                                message = (
                                    f"*Potent Token*\n\n"
                                    f"*Name:* {token['baseToken']['name']}\n"
                                    f"*Symbol:* {token['baseToken']['symbol']}\n"
                                    f"*URL:* {token['url']}\n"
                                    f"*Predictions mean probability:* {mean_preds * 100:.2f}%\n"
                                    f"RF: {rf_proba * 100:.2f}%\n"
                                    f"xGB: {gb_proba * 100:.2f}%\n"
                                    f"LR: {lr_proba * 100:.2f}%\n"
                                    f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                                )
                                await send_telegram_message(session, message, CHAT_IDs)

                                if mean_preds >= .75:
                                    message = (
                                        f"🔔🔔🔔*POTĘŻNY TOKEN (dla jego)!*🔔🔔🔔\n\n"
                                        f"*Name:* {token['baseToken']['name']}\n"
                                        f"*Symbol:* {token['baseToken']['symbol']}\n"
                                        f"*URL:* {token['url']}\n"
                                        f"*Predictions mean probability:* {mean_preds * 100:.2f}%\n"
                                        f"RF: {rf_proba * 100:.2f}%\n"
                                        f"xGB: {gb_proba * 100:.2f}%\n"
                                        f"LR: {lr_proba * 100:.2f}%\n"
                                        f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                                    )
                                    await send_telegram_message(session, message, CHAT_IDs)
                                    webbrowser.open(token["url"])
                                    for _ in range(5): beep(sound='coin')
                else:
                    pass
                    # Możesz dodać logowanie braku ważnych par, jeśli chcesz

            # Czekaj przed kolejnym sprawdzeniem
            await asyncio.sleep(10)  # Zmieniono na 60 sekund dla praktyczności
            tas_dict = fetch_ta(client, 'SOLUSDT', ITVS) | fetch_ta(client, 'BTCUSDT', ITVS)


if __name__ == "__main__":
    asyncio.run(main())
