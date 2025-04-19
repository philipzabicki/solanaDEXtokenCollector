import os
import csv
import torch
import math
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from datetime import timedelta, time, datetime, timezone
import pytz
from math import exp
import joblib
import aiohttp
import asyncio
import webbrowser
from beepy import beep
from new_pool_sniper import (
    ITVS,
    fetch_valid_pairs_details,
    fetch_new_tokens,
    check_credentials
)
from credentials import *
from binance.um_futures import UMFutures
from tenacity import retry, stop_after_attempt, wait_exponential
from collections import deque
from typing import Tuple, List
from common import fetch_ta_from_config

# Additional function to save prediction results to a CSV file
def save_prediction_record(record, models_mid_date):
    os.makedirs("preds", exist_ok=True)
    preds_file = f"preds/{models_mid_date.strftime('%Y%m%d_%H%M%S')}.csv"
    file_exists = os.path.isfile(preds_file)
    with open(preds_file, "a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

# Function to sort predictions.csv file at startup
def sort_predictions_file(preds_file: str = "preds/predictions.csv"):
    if os.path.exists(preds_file):
        df_preds = pd.read_csv(preds_file, encoding='utf-8')
        # Konwersja wartości z tekstu (usuwamy znak '%' dla cls_prediction)
        df_preds["cls_prediction_float"] = df_preds["cls_prediction"].str.rstrip('%').astype(float)
        df_preds["reg_prediction_float"] = df_preds["reg_prediction"].astype(float)
        # Sortowanie malejąco - najpierw wg cls_prediction, potem wg reg_prediction
        df_preds = df_preds.sort_values(by=["cls_prediction_float", "reg_prediction_float"], ascending=False)
        # Usuwamy pomocnicze kolumny
        df_preds = df_preds.drop(columns=["cls_prediction_float", "reg_prediction_float"])
        df_preds.to_csv(preds_file, index=False, encoding="utf-8")
        print("Predictions file sorted.")

LAUNCH_TIME = timedelta(minutes=0)
SKIP_TIME = timedelta(minutes=10)

CLS_PRECISION = 1/3
CLS_THRESHOLD_1 = 0.5
CLS_THRESHOLD_2 = 0.8
REG_THRESHOLD_1 = 4.67
REG_THRESHOLD_2 = 5.15
# REG_95_PERC = 4.4454452
# REG_100_PERC = 6.090765955

CHAT_IDs = ["5011277677", "7228159263"]

CLS_MODEL_FULLPATH = 'models/final_cls_model.joblib'
REG_MODEL_FULLPATH = 'models/final_reg_model.joblib'

EMBED_MODEL = 'mixedbread-ai/mxbai-embed-large-v1'
NAME_INDICES_PATH = "models/important_name_indices.joblib"
SYMBOL_INDICES_PATH = "models/important_symbol_indices.joblib"
# TOKENIZED_NEME_DIM = 128
# TOKENIZED_SYMBOL_DIM = 64

TA_CONFIG = {
    "BTCUSDT": {
        "ADOSC": ["1m", "5m", "15m", "1h", "4h"],
        "OBV": ["1m", "5m", "15m"],
        "ATR": ["1m", "5m", "15m"],
        "RSI": ["1m", "5m", "15m", "1h", "4h"],
        "ULTOSC": ["1m", "5m", "15m", "1h", "4h", "1d"]
    },
    "SOLUSDT": {
        "ADOSC": ["1m", "5m", "15m", "1h", "4h"],
        "OBV": ["1m", "5m", "15m"],
        "ATR": ["1m", "5m", "15m"],
        "RSI": ["1m", "5m", "15m", "1h", "4h"],
        "ULTOSC": ["1m", "5m", "15m", "1h", "4h", "1d"]
    }
}

COLS_TO_DROP = [
    "worthy",
    "pairCreatedAt"
]

MAX_CONCURRENT_REQUESTS = 30  # Limit równoległych zapytań
SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

DEX_MAPPING = {
    "raydium": 6,
    "fluxbeam": 5,
    "meteora": 4,
    "pumpswap": 3,
    "orca": 2,
    "pumpfunamm": 1
}
# ======================================================================
# SESSION FEATURE SETUP
# ======================================================================
sessions = {
    # Ameryka Północna
    'NYSE': {'start': time(9, 30), 'end': time(16, 0), 'timezone': 'America/New_York'},
    'NASDAQ': {'start': time(9, 30), 'end': time(16, 0), 'timezone': 'America/New_York'},
    
    # Europa
    'LSE': {'start': time(8, 0), 'end': time(16, 30), 'timezone': 'Europe/London'},
    'Xetra': {'start': time(9, 0), 'end': time(17, 30), 'timezone': 'Europe/Berlin'},
    
    # Azja i Pacyfik
    'TSE_Morning': {'start': time(9, 0), 'end': time(11, 30), 'timezone': 'Asia/Tokyo'},
    'TSE_Afternoon': {'start': time(12, 30), 'end': time(15, 30), 'timezone': 'Asia/Tokyo'},
    'SSE_Morning': {'start': time(9, 30), 'end': time(11, 30), 'timezone': 'Asia/Shanghai'},
    'SSE_Afternoon': {'start': time(13, 0), 'end': time(15, 0), 'timezone': 'Asia/Shanghai'},
    'BSE': {'start': time(9, 15), 'end': time(15, 30), 'timezone': 'Asia/Kolkata'},
    'ASX': {'start': time(10, 0), 'end': time(16, 0), 'timezone': 'Australia/Sydney'},
    'HOSE_Morning': {'start': time(9, 15), 'end': time(11, 30), 'timezone': 'Asia/Ho_Chi_Minh'},
    'HOSE_Afternoon': {'start': time(13, 0), 'end': time(14, 30), 'timezone': 'Asia/Ho_Chi_Minh'},
    'PSE_Morning': {'start': time(9, 30), 'end': time(12, 0), 'timezone': 'Asia/Manila'},
    'PSE_Afternoon': {'start': time(13, 0), 'end': time(14, 45), 'timezone': 'Asia/Manila'},
    'PSX': {'start': time(9, 32), 'end': time(15, 30), 'timezone': 'Asia/Karachi'},
    'SET_Morning': {'start': time(10, 0), 'end': time(12, 30), 'timezone': 'Asia/Bangkok'},
    'SET_Afternoon': {'start': time(14, 0), 'end': time(16, 30), 'timezone': 'Asia/Bangkok'},
    'IDX': {'start': time(9, 0), 'end': time(15, 50), 'timezone': 'Asia/Jakarta'},
    
    # Bliski Wschód i Afryka
    'DFM': {'start': time(10, 0), 'end': time(15, 0), 'timezone': 'Asia/Dubai'},
    'NSE_Nigeria': {'start': time(10, 0), 'end': time(14, 20), 'timezone': 'Africa/Lagos'},
    'BIST_Morning': {'start': time(9, 30), 'end': time(12, 30), 'timezone': 'Europe/Istanbul'},
    'BIST_Afternoon': {'start': time(14, 0), 'end': time(17, 30), 'timezone': 'Europe/Istanbul'},
    'NSE_Kenya': {'start': time(9, 0), 'end': time(15, 0), 'timezone': 'Africa/Nairobi'},

    # Ameryka Południowa
    'B3': {'start': time(10, 0), 'end': time(16, 55), 'timezone': 'America/Sao_Paulo'},
    'BCBA': {'start': time(11, 0), 'end': time(17, 0), 'timezone': 'America/Argentina/Buenos_Aires'},
    'BVC': {'start': time(9, 30), 'end': time(15, 55), 'timezone': 'America/Bogota'},
}

def compute_session_feature(row, session_info):
    # Get the timestamp from the 'pairCreatedAt' column and ensure it's tz-aware.
    timestamp = row['pairCreatedAt']
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    
    # Convert timestamp to the session's local timezone.
    tz = pytz.timezone(session_info['timezone'])
    timestamp_local = timestamp.astimezone(tz)
    
    session_start_time = session_info['start']
    session_end_time = session_info['end']
    
    # Create datetime objects for the session's start and end for the current day.
    start_datetime = timestamp_local.replace(hour=session_start_time.hour,
                                               minute=session_start_time.minute,
                                               second=session_start_time.second,
                                               microsecond=0)
    end_datetime = timestamp_local.replace(hour=session_end_time.hour,
                                             minute=session_end_time.minute,
                                             second=session_end_time.second,
                                             microsecond=0)
    
    # Case 1: Timestamp is before session open (pre-session break)
    if timestamp_local < start_datetime:
        # Previous session's closing is assumed to be on the previous day.
        prev_day = timestamp_local - timedelta(days=1)
        # Build the datetime for previous session's close on the previous day.
        prev_close = start_datetime.replace(year=prev_day.year,
                                              month=prev_day.month,
                                              day=prev_day.day,
                                              hour=session_end_time.hour,
                                              minute=session_end_time.minute,
                                              second=session_end_time.second,
                                              microsecond=0)
        # Total duration of the break (in seconds) between previous close and today's open.
        off_period = (start_datetime - prev_close).total_seconds()
        elapsed = (timestamp_local - prev_close).total_seconds()
        # Normalize the elapsed time in the break period.
        x = elapsed / off_period
        # Use a non-linear sinusoidal transformation: 0 at boundaries, -1 at the middle.
        feature = -math.sin(math.pi * x)
        return feature

    # Case 2: Timestamp is after session close (post-session break)
    elif timestamp_local > end_datetime:
        # Next session's open is assumed to be on the following day.
        next_day = timestamp_local + timedelta(days=1)
        next_open = start_datetime.replace(year=next_day.year,
                                           month=next_day.month,
                                           day=next_day.day,
                                           hour=session_start_time.hour,
                                           minute=session_start_time.minute,
                                           second=session_start_time.second,
                                           microsecond=0)
        off_period = (next_open - end_datetime).total_seconds()
        elapsed = (timestamp_local - end_datetime).total_seconds()
        x = elapsed / off_period
        feature = -math.sin(math.pi * x)
        return feature

    # Case 3: Timestamp is within the session time.
    else:
        session_period = (end_datetime - start_datetime).total_seconds()
        elapsed = (timestamp_local - start_datetime).total_seconds()
        x = elapsed / session_period
        # Sinusoidal transformation: 0 at open (x=0) and close (x=1), +1 at the middle (x=0.5).
        feature = math.sin(math.pi * x)
        return feature


# Circuit Breaker
class CircuitBreaker:
    def __init__(self, max_failures=3, reset_timeout=60):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure = None

    def __call__(self, func):
        async def wrapper(*args, **kwargs):
            if (
                self.last_failure
                and (datetime.now() - self.last_failure).seconds < self.reset_timeout
            ):
                raise Exception("Circuit breaker blocked")
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                self.failures += 1
                self.last_failure = datetime.now()
                if self.failures >= self.max_failures:
                    print(f"Circuit breaker triggered! Waiting {self.reset_timeout}s")
                    await asyncio.sleep(self.reset_timeout)
                raise

        return wrapper


# Walidacja danych
def validate_pair_structure(pair: dict) -> bool:
    required_keys = {
        "pairAddress",
        "baseToken",
        "priceUsd",
        "txns",
        "volume",
        "liquidity",
        "fdv",
        "pairCreatedAt",
    }
    return all(key in pair for key in required_keys)


# Retry dla pojedynczych adresów
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=120))
@CircuitBreaker(max_failures=100, reset_timeout=60)
async def fetch_single_address(session: aiohttp.ClientSession, address: str) -> dict:
    async with SEMAPHORE:
        url = f"https://api.dexscreener.com/latest/dex/pairs/solana/{address}"
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                data = await response.json()
                if data.get("pairs") and validate_pair_structure(data["pairs"][0]):
                    return data["pairs"][0]
            return None


# Główna funkcja z pełną obsługą błędów
async def fetch_valid_pairs_details(
    session: aiohttp.ClientSession, address_list: List[str], verbose: bool = True
) -> Tuple[List[dict], List[str]]:
    tasks = [fetch_single_address(session, addr) for addr in address_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    valid_pairs = []
    failed_addresses = []
    now = datetime.now(timezone.utc)

    for addr, result in zip(address_list, results):
        if isinstance(result, Exception) or result is None:
            if verbose:
                print(f"Failed to fetch {addr}: {str(result)}")
            failed_addresses.append(addr)
            continue

        creation_time = datetime.fromtimestamp(
            result["pairCreatedAt"] / 1000, tz=timezone.utc
        )
        if SKIP_TIME > (now - creation_time) > LAUNCH_TIME:
            valid_pairs.append(result)
            if verbose:
                print(f'Valid pair address: {result["pairAddress"]}')

    return valid_pairs, failed_addresses


async def get_input(df, scaler, embed_model, name_indices=None, symbol_indices=None):
    # print(f'name_indices: {name_indices}')
    # print(f'symbol_indices: {symbol_indices}')
    token_names = df["baseTokenName"].astype(str).str.lower().fillna('').tolist()
    token_symbols = df["baseTokenSymbol"].astype(str).str.lower().fillna('').tolist()

    full_names_embeddings = embed_model.encode(token_names, show_progress_bar=False, convert_to_numpy=True)
    full_symbols_embeddings = embed_model.encode(token_symbols, show_progress_bar=False, convert_to_numpy=True)
    # print(f'full_names_embeddings.shape: {full_names_embeddings.shape}')
    # print(f'full_symbols_embeddings.shape: {full_symbols_embeddings.shape}')
    # reduced_names_embeddings = full_names_embeddings[:, :TOKENIZED_NEME_DIM]
    # reduced_symbols_embeddings = full_symbols_embeddings[:, :TOKENIZED_SYMBOL_DIM]
    reduced_names_embeddings = full_names_embeddings[:, name_indices]
    reduced_symbols_embeddings = full_symbols_embeddings[:, symbol_indices]

    names_vectors = pd.DataFrame(reduced_names_embeddings, index=df.index, columns=[f"nameVectorSelected_DimIdx{i}" for i in name_indices])
    symbols_vectors = pd.DataFrame(reduced_symbols_embeddings, index=df.index, columns=[f"symbolVectorSelected_DimIdx{i}" for i in symbol_indices])

    df.drop(columns=["baseTokenName", "baseTokenSymbol"], inplace=True)
    ret = pd.concat([df, names_vectors, symbols_vectors], axis=1)

    if scaler is None:
        # If no scaler is provided, return the DataFrame as is
        ret = ret.reindex(sorted(ret.columns), axis=1)
        return ret.drop(columns=COLS_TO_DROP)
    else:
        # Transform the DataFrame using the provided scaler
        return pd.DataFrame(scaler.transform(ret), columns=ret.columns).drop(
            columns=COLS_TO_DROP
        ).reindex(sorted(df.columns), axis=1)


async def get_features_df(detail: dict, tas_dict: dict = None) -> pd.DataFrame:
    # Reduce model calls by filtering
    if "m5" not in detail["priceChange"]:
        print(f'Pair {detail["pairAddress"]} has no m5 price change yet.')
        return None
    elif detail["liquidity"]["usd"] == 0:
        print(f'Pair {detail["pairAddress"]} has low liquidity: {detail["liquidity"]["usd"]} USD.')
        return None
    try:
        det_dict = {
            "dexId": DEX_MAPPING[detail["dexId"]],
            "baseTokenName": detail["baseToken"]["name"],
            "baseTokenSymbol": detail["baseToken"]["symbol"],
            "priceNative": float(detail["priceNative"]),
            "priceUsd": float(detail["priceUsd"]),
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
            "fdv": (
                detail["fdv"] if "fdv" in detail else 0.0
            ),  # Check if 'fdv' exists in detail
        }
    except KeyError as e:
        print(f'Failed creation of features for {detail["baseToken"]["name"]} pair address {detail["pairAddress"]} ({e}).')
        # print(f'Pair details: {detail}')
        return None
        # return pd.DataFrame()

    # Calculate liquidity to FDV ratio, set to 0 if conditions are not met
    liq = {
        "worthy": -1,
        "liq_fdv_ratio": (
            detail["liquidity_usd"] / detail["fdv"]
            if "liquidity_usd" in detail and detail["fdv"] > 0
            else 0.0
        ),
    }

    # Fetch technical analysis data if not provided
    if tas_dict is None:
        client = UMFutures(binance_API_KEY, binance_SECRET_KEY)
        tas_dict = fetch_ta_from_config(client, TA_CONFIG)

    # Update the details dictionary with technical analysis and liquidity data
    det_dict.update(tas_dict | liq)
    df = pd.DataFrame([det_dict])
    
    # --- Dodajemy nowe cechy sesyjne na podstawie pairCreatedAt ---
    # Konwersja wartości pairCreatedAt (ms) do datetime (tz-aware)
    df["pairCreatedAt"] = pd.to_datetime(detail["pairCreatedAt"], unit="ms", utc=True)
    df["hour"] = df["pairCreatedAt"].dt.hour
    df["weekday"] = df["pairCreatedAt"].dt.weekday  # Monday = 0, Sunday = 6
    # Obliczenie i dodanie kolumn sesyjnych
    for session_name, session_info in sessions.items():
        feature_name = f"session_{session_name}_feature"
        df[feature_name] = df.apply(lambda row: compute_session_feature(row, session_info), axis=1)
    # -----------------------------------------------------------------

    return df.reindex(sorted(df.columns), axis=1)


async def send_telegram_message(session, message: str, chat_ids: list):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for chat_id in chat_ids:
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",  # Opcjonalnie, możesz użyć innego formatu
        }
        async with session.post(url, data=payload) as response:
            if response.status != 200:
                print(f"Failed to send message to {chat_id}: {response.status}")
                resp_text = await response.text()
                print(resp_text)


async def main():
    sort_predictions_file("preds/predictions.csv")
    if torch.cuda.is_available():
        device = "cuda"
    else:
        raise ValueError('CUDA not available!')
    
    processor = {
        "seen_addresses": deque(maxlen=1_000),
        "failed_addresses": deque(maxlen=15),
    }
    check_credentials(binance_API_KEY, binance_SECRET_KEY)

    NAME_INDICES = joblib.load(NAME_INDICES_PATH)
    SYMBOL_INDICES = joblib.load(SYMBOL_INDICES_PATH)
    print(f'NAME_INDICES: {NAME_INDICES} typ: {type(NAME_INDICES)}')
    print(f'SYMBOL_INDICES: {SYMBOL_INDICES} typ: {type(SYMBOL_INDICES)}')
    # Load models
    embed_model = SentenceTransformer(EMBED_MODEL, device=device)
    cls_model = joblib.load(CLS_MODEL_FULLPATH)
    reg_model = joblib.load(REG_MODEL_FULLPATH)

    # Get model file creation dates
    cls_model_date = datetime.fromtimestamp(os.path.getmtime(CLS_MODEL_FULLPATH), tz=timezone.utc)
    reg_model_date = datetime.fromtimestamp(os.path.getmtime(REG_MODEL_FULLPATH), tz=timezone.utc)

    mid_timestamp = (cls_model_date.timestamp() + reg_model_date.timestamp()) / 2
    mid_model_date = datetime.fromtimestamp(mid_timestamp, tz=timezone.utc)

    async with aiohttp.ClientSession() as session:
        client = UMFutures(binance_API_KEY, binance_SECRET_KEY)
        tas_dict = fetch_ta_from_config(client, TA_CONFIG)

        while True:
            processor["failed_addresses"] = deque(
                dict.fromkeys(
                    addr
                    for addr in processor["failed_addresses"]
                    if addr not in set(processor["seen_addresses"])
                ),
                maxlen=15,
            )
            # print(f'processor["failed_addresses"]: {processor["failed_addresses"]}')
            # new_tokens = await fetch_new_tokens(session)
            # addresses = list(set(token['attributes']['address'] for token in new_tokens) - set(seen_addresses))
            new_tokens = await fetch_new_tokens(session)
            current_addresses = [t["attributes"]["address"] for t in new_tokens]
            all_addresses = list(
                set(current_addresses + list(processor["failed_addresses"]))
            )
            # print(f'new_tokens: {new_tokens}')

            valid_pairs, failed_pairs = await fetch_valid_pairs_details(
                session, all_addresses, verbose=False
            )
            processor["failed_addresses"].extend(failed_pairs)
            for pair in valid_pairs:
                if pair["pairAddress"] not in processor["seen_addresses"]:
                    try:
                        features_df = await get_features_df(pair, tas_dict)
                        # print(f'features_df.columns {features_df.columns}')
                        if features_df is None:
                            processor["failed_addresses"].append(pair["pairAddress"])
                        else:
                            # Dodaj do seen_addresses DOPIERO po udanym przetworzeniu
                            processor["seen_addresses"].append(pair["pairAddress"])
                            input_data = await get_input(df=features_df.copy(), scaler=None, embed_model=embed_model, name_indices=NAME_INDICES, symbol_indices=SYMBOL_INDICES)
                            cls_proba = cls_model.predict_proba(input_data)[0][1]
                            reg_proba = reg_model.predict(input_data)[0]

                            cls_proba_adj = cls_proba * CLS_PRECISION

                            # Create prediction record
                            record = {
                                "prediction_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                                "cls_prediction": f"{cls_proba*100:.2f}%",
                                "cls_prediction_adjusted": f"{cls_proba_adj*100:.2f}%",
                                "reg_prediction": f"{reg_proba:.6f}",
                                "token_name": pair["baseToken"]["name"],
                                "token_symbol": pair["baseToken"]["symbol"],
                                "url": pair["url"],
                                "cls_model_creation": cls_model_date.strftime("%Y-%m-%d %H:%M:%S"),
                                "reg_model_creation": reg_model_date.strftime("%Y-%m-%d %H:%M:%S")
                            }

                            if (cls_proba >= CLS_THRESHOLD_2) or (reg_proba >= REG_THRESHOLD_2):
                                    print("##################################################")
                                    print(f"🔔🔔🔔 POTĘŻNY TOKEN (dla jego)! 🔔🔔🔔")
                                    print(f"Classification predition: {cls_proba * 100:.2f}%  (adjusted {cls_proba_adj* 100:.2f}%)")
                                    print(f"Regression predition: {reg_proba:.6f}  (suggested per Sol stake: {(reg_proba*cls_proba)*0.01:.6f} Sol)\n")
                                    print(f'Name: {pair["baseToken"]["name"]} Symbol: {pair["baseToken"]["symbol"]}')
                                    print(f'url: {pair["url"]}')
                                    print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}')
                                    print("##################################################")
                                    message = (
                                        f"🔔🔔🔔*POTĘŻNY TOKEN (dla jego)!*🔔🔔🔔\n\n"
                                        f"*Classification predition:* {cls_proba * 100:.2f}% (adjusted {cls_proba_adj* 100:.2f}%)\n"
                                        f"*Regression predition:* {reg_proba:.6f}  (suggested per Sol stake: {(reg_proba*cls_proba)*0.01:.6f} Sol)\n"
                                        f"*Name:* {pair['baseToken']['name']}\n"
                                        f"*Symbol:* {pair['baseToken']['symbol']}\n"
                                        f"*URL:* {pair['url']}\n"
                                        f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                                    )
                                    await send_telegram_message(
                                        session, message, CHAT_IDs
                                    )
                                    webbrowser.open(pair["url"])
                                    for _ in range(5):
                                        beep(sound="coin")
                            elif (cls_proba >= CLS_THRESHOLD_1) or (reg_proba >= REG_THRESHOLD_1):
                                for _ in range(1):
                                    beep(sound="coin")
                                print("##################################################")
                                print(f"Classification predition: {cls_proba * 100:.2f}%  (adjusted {cls_proba_adj* 100:.2f}%)")
                                print(f"Regression predition: {reg_proba:.6f}  (suggested per Sol stake: {(reg_proba*cls_proba)*0.01:.6f} Sol)\n")
                                print(f'Name: {pair["baseToken"]["name"]} Symbol: {pair["baseToken"]["symbol"]}')
                                print(f'url: {pair["url"]}')
                                print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}')
                                print("##################################################")

                                message = (
                                    f"*Potent Token*\n\n"
                                    f"*Classification predition:* {cls_proba * 100:.2f}% (adjusted {cls_proba_adj* 100:.2f}%)\n"
                                    f"*Regression predition:* {reg_proba:.6f}  (suggested per Sol stake: {(reg_proba*cls_proba)*0.01:.6f} Sol)\n"
                                    f"*Name:* {pair['baseToken']['name']}\n"
                                    f"*Symbol:* {pair['baseToken']['symbol']}\n"
                                    f"*URL:* {pair['url']}\n"
                                    f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                                )
                                await send_telegram_message(session, message, CHAT_IDs)
                                webbrowser.open(pair["url"])
                            # Save all predictions regardless of classification and regression results
                            save_prediction_record(record, mid_model_date)

                    except Exception as e:
                        print(f"Processing failed for {pair['pairAddress']}: {str(e)}")
                        processor["failed_addresses"].append(pair["pairAddress"])

            # Czekaj przed kolejnym sprawdzeniem
            await asyncio.sleep(1)
            tas_dict = fetch_ta_from_config(client, TA_CONFIG)


if __name__ == "__main__":
    asyncio.run(main())
