import os
import csv
import torch
import json

# import lightgbm as lgb
# lgb.register_logger(lambda msg: None)
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from datetime import timedelta, time, datetime, timezone
import joblib
import aiohttp
import asyncio
import webbrowser

# from beepy import beep
from credentials import *
from binance.um_futures import UMFutures
from tenacity import retry, stop_after_attempt, wait_exponential
from collections import deque
from typing import Tuple, List
from common import fetch_ta_from_config, check_credentials, fetch_new_tokens
from credentials import TELEGRAM_CHAT_IDs, TELEGRAM_TOKEN


LAUNCH_TIME = timedelta(minutes=0)
SKIP_TIME = timedelta(minutes=10)

CLS_PRECISION = 0.0062  # final model test set precision
CLS_THRESHOLD_1 = 0.5
CLS_THRESHOLD_2 = 0.8

CLS_MODEL_FULLPATH = "models/final_cls_model.joblib"
REG_MODEL_FULLPATH = "models/final_reg_model.joblib"

EMBED_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

NAME_PCA_PATH = "models/pca_name.joblib"
SYMBOL_PCA_PATH = "models/pca_symbol.joblib"
NAME_INDICES_PATH = "models/important_name_pca_indices.joblib"
SYMBOL_INDICES_PATH = "models/important_symbol_pca_indices.joblib"

TA_CONFIG_PATH = "models/selected_indicators.json"
FEATURE_MEDIANS_PATH = "models/feature_medians.joblib"

COLS_TO_DROP = [
    "worthy",
    "pairCreatedAt",
    "fdv",
    # "liq_fdv_ratio"
]

MAX_CONCURRENT_REQUESTS = 30  # Limit równoległych zapytań
SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

DEX_MAPPING = {
    "pumpswap": 0.33975964518957025,
    "raydium": 0.26431167911030706,
    "meteora": 0.19479446847594512,
    "fluxbeam": 0.18783777194653165,
    "pumpfun": 0.006137734829555462,
    "heaven": 0.0027365062874963554,
    "launchlab": 0.0025703257238556677,
    "orca": 0.0017055897370068193,
    "meteoradbc": 0.00012836702221346541,
    "pumpfunamm": 9.950931954532202e-06,
    "tokenmill": 5.970559172719321e-06,
    "dexlab": 1.9901863909064404e-06,
}

# SESSION FEATURE SETUP
sessions = {
    # Ameryka Północna
    "NYSE": {"start": time(9, 30), "end": time(16, 0), "timezone": "America/New_York"},
    "NASDAQ": {
        "start": time(9, 30),
        "end": time(16, 0),
        "timezone": "America/New_York",
    },
    # Europa
    "LSE": {"start": time(8, 0), "end": time(16, 30), "timezone": "Europe/London"},
    "Xetra": {"start": time(9, 0), "end": time(17, 30), "timezone": "Europe/Berlin"},
    # Azja i Pacyfik
    "TSE_Morning": {"start": time(9, 0), "end": time(11, 30), "timezone": "Asia/Tokyo"},
    "TSE_Afternoon": {
        "start": time(12, 30),
        "end": time(15, 30),
        "timezone": "Asia/Tokyo",
    },
    "SSE_Morning": {
        "start": time(9, 30),
        "end": time(11, 30),
        "timezone": "Asia/Shanghai",
    },
    "SSE_Afternoon": {
        "start": time(13, 0),
        "end": time(15, 0),
        "timezone": "Asia/Shanghai",
    },
    "BSE": {"start": time(9, 15), "end": time(15, 30), "timezone": "Asia/Kolkata"},
    "ASX": {"start": time(10, 0), "end": time(16, 0), "timezone": "Australia/Sydney"},
    "HOSE_Morning": {
        "start": time(9, 15),
        "end": time(11, 30),
        "timezone": "Asia/Ho_Chi_Minh",
    },
    "HOSE_Afternoon": {
        "start": time(13, 0),
        "end": time(14, 30),
        "timezone": "Asia/Ho_Chi_Minh",
    },
    "PSE_Morning": {
        "start": time(9, 30),
        "end": time(12, 0),
        "timezone": "Asia/Manila",
    },
    "PSE_Afternoon": {
        "start": time(13, 0),
        "end": time(14, 45),
        "timezone": "Asia/Manila",
    },
    "PSX": {"start": time(9, 32), "end": time(15, 30), "timezone": "Asia/Karachi"},
    "SET_Morning": {
        "start": time(10, 0),
        "end": time(12, 30),
        "timezone": "Asia/Bangkok",
    },
    "SET_Afternoon": {
        "start": time(14, 0),
        "end": time(16, 30),
        "timezone": "Asia/Bangkok",
    },
    "IDX": {"start": time(9, 0), "end": time(15, 50), "timezone": "Asia/Jakarta"},
    # Bliski Wschód i Afryka
    "DFM": {"start": time(10, 0), "end": time(15, 0), "timezone": "Asia/Dubai"},
    "NSE_Nigeria": {
        "start": time(10, 0),
        "end": time(14, 20),
        "timezone": "Africa/Lagos",
    },
    "BIST_Morning": {
        "start": time(9, 30),
        "end": time(12, 30),
        "timezone": "Europe/Istanbul",
    },
    "BIST_Afternoon": {
        "start": time(14, 0),
        "end": time(17, 30),
        "timezone": "Europe/Istanbul",
    },
    "NSE_Kenya": {
        "start": time(9, 0),
        "end": time(15, 0),
        "timezone": "Africa/Nairobi",
    },
    # Ameryka Południowa
    "B3": {"start": time(10, 0), "end": time(16, 55), "timezone": "America/Sao_Paulo"},
    "BCBA": {
        "start": time(11, 0),
        "end": time(17, 0),
        "timezone": "America/Argentina/Buenos_Aires",
    },
    "BVC": {"start": time(9, 30), "end": time(15, 55), "timezone": "America/Bogota"},
}


def compute_session_feature(ts_utc, si):
    ts = pd.Timestamp(ts_utc).tz_convert(si["timezone"])

    start = si["start"]
    end = si["end"]
    if hasattr(start, "time"):
        start = start.time()
    if hasattr(end, "time"):
        end = end.time()

    start_td = pd.Timedelta(
        hours=start.hour, minutes=start.minute, seconds=start.second
    )
    end_td = pd.Timedelta(hours=end.hour, minutes=end.minute, seconds=end.second)

    day = ts.normalize()
    crosses = end <= start

    if crosses:
        # "rano" należy do sesji rozpoczętej dzień wcześniej
        base = day - pd.Timedelta(days=1) if ts.time() < end else day
        start_dt = base + start_td
        end_dt = base + pd.Timedelta(days=1) + end_td
        prev_close = base + end_td
        next_open = base + pd.Timedelta(days=1) + start_td
    else:
        base = day
        start_dt = base + start_td
        end_dt = base + end_td
        prev_close = base - pd.Timedelta(days=1) + end_td
        next_open = base + pd.Timedelta(days=1) + start_td

    if ts < start_dt:
        off = (start_dt - prev_close).total_seconds()
        x = (ts - prev_close).total_seconds() / off
        return float(np.float32(-np.sin(np.pi * x)))

    if ts > end_dt:
        off = (next_open - end_dt).total_seconds()
        x = (ts - end_dt).total_seconds() / off
        return float(np.float32(-np.sin(np.pi * x)))

    sess = (end_dt - start_dt).total_seconds()
    x = (ts - start_dt).total_seconds() / sess
    return float(np.float32(np.sin(np.pi * x)))


# Additional function to save prediction results to a CSV file
def save_prediction_record(record, models_mid_date):
    os.makedirs("data/preds", exist_ok=True)
    preds_file = f"data/preds/{models_mid_date.strftime('%Y%m%d_%H%M%S')}.csv"
    file_exists = os.path.isfile(preds_file)
    with open(preds_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


# Function to sort predictions.csv file at startup
def sort_predictions_file(preds_file: str = "data/preds/predictions.csv"):
    if os.path.exists(preds_file):
        df_preds = pd.read_csv(preds_file, encoding="utf-8")
        # Konwersja wartości z tekstu (usuwamy znak '%' dla cls_prediction)
        df_preds["cls_prediction_float"] = (
            df_preds["cls_prediction"].str.rstrip("%").astype(float)
        )
        df_preds["reg_prediction_float"] = df_preds["reg_prediction"].astype(float)
        # Sortowanie malejąco - najpierw wg cls_prediction, potem wg reg_prediction
        df_preds = df_preds.sort_values(
            by=["cls_prediction_float", "reg_prediction_float"], ascending=False
        )
        # Usuwamy pomocnicze kolumny
        df_preds = df_preds.drop(
            columns=["cls_prediction_float", "reg_prediction_float"]
        )
        df_preds.to_csv(preds_file, index=False, encoding="utf-8")
        print("Predictions file sorted.")


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

        # creation_time = datetime.fromtimestamp(
        #     result["pairCreatedAt"] / 1000, tz=timezone.utc
        # )
        # if SKIP_TIME > (now - creation_time) > LAUNCH_TIME:
        #     valid_pairs.append(result)
        valid_pairs.append(result)
        if verbose:
            print(f'Valid pair address: {result["pairAddress"]}')

    return valid_pairs, failed_addresses


async def get_input(
    df,
    scaler,
    embed_model,
    name_pca,
    symbol_pca,
    name_indices=None,
    symbol_indices=None,
):
    # print(f'name_indices: {name_indices}')
    # print(f'symbol_indices: {symbol_indices}')
    token_names = df["baseTokenName"].astype(str).str.lower().fillna("").tolist()
    token_symbols = df["baseTokenSymbol"].astype(str).str.lower().fillna("").tolist()

    texts = token_names + token_symbols
    emb = embed_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    n = len(token_names)
    full_names_embeddings = emb[:n]
    full_symbols_embeddings = emb[n:]

    names_pca = name_pca.transform(full_names_embeddings).astype(np.float32)
    symbols_pca = symbol_pca.transform(full_symbols_embeddings).astype(np.float32)

    reduced_names_embeddings = names_pca[:, name_indices]
    reduced_symbols_embeddings = symbols_pca[:, symbol_indices]

    names_vectors = pd.DataFrame(
        reduced_names_embeddings,
        index=df.index,
        columns=[f"namePCA_Selected_Dim{i}" for i in name_indices],
    )
    symbols_vectors = pd.DataFrame(
        reduced_symbols_embeddings,
        index=df.index,
        columns=[f"symbolPCA_Selected_Dim{i}" for i in symbol_indices],
    )

    df.drop(columns=["baseTokenName", "baseTokenSymbol"], inplace=True)
    ret = pd.concat([df, names_vectors, symbols_vectors], axis=1)

    if scaler is None:
        # If no scaler is provided, return the DataFrame as is
        ret = ret.reindex(sorted(ret.columns), axis=1)
        return ret.drop(columns=COLS_TO_DROP, errors="ignore")
    else:
        ret = pd.DataFrame(scaler.transform(ret), columns=ret.columns)
        ret = ret.drop(columns=COLS_TO_DROP, errors="ignore")
        return ret.reindex(sorted(ret.columns), axis=1)


async def get_features_df(detail, client, ta_config, ta_cache):
    # szybkie filtry
    if "priceChange" not in detail or "m5" not in detail["priceChange"]:
        return None
    if detail.get("liquidity", {}).get("usd", 0) == 0:
        return None

    try:
        det_dict = {
            "dexId": DEX_MAPPING.get(detail.get("dexId"), 0.0),
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
            "txns_h24_buys": detail["txns"]["h24"]["buys"],
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
            "fdv": float(detail.get("fdv", 0.0)),
        }
    except KeyError:
        return None

    liq = {
        "worthy": -1,
        "liq_fdv_ratio": (
            (det_dict["liquidity_usd"] / det_dict["fdv"])
            if det_dict["fdv"] > 0
            else 0.0
        ),
    }

    pair_ms = int(detail["pairCreatedAt"])
    tas_dict = await asyncio.to_thread(
        fetch_ta_from_config,
        client,
        ta_config,
        100,  # klines_limit
        pair_ms,  # ref_ms
        ta_cache,  # cache
    )

    det_dict.update(liq)
    det_dict.update(tas_dict)

    df = pd.DataFrame([det_dict])

    # cechy czasowe + sesje (bez df.apply)
    ts = pd.to_datetime(detail["pairCreatedAt"], unit="ms", utc=True)
    df["pairCreatedAt"] = ts
    df["hour"] = ts.hour
    df["weekday"] = ts.weekday()

    for session_name, si in sessions.items():
        df[f"session_{session_name}_feature"] = compute_session_feature(ts, si)

    return df


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
    if torch.cuda.is_available():
        device = "cuda"
    else:
        raise ValueError("CUDA not available!")

    processor = {
        "seen_addresses": deque(maxlen=1_000),
        "failed_addresses": deque(maxlen=15),
    }

    check_credentials(binance_API_KEY, binance_SECRET_KEY)

    NAME_INDICES = joblib.load(NAME_INDICES_PATH)
    SYMBOL_INDICES = joblib.load(SYMBOL_INDICES_PATH)
    NAME_PCA = joblib.load(NAME_PCA_PATH)
    SYMBOL_PCA = joblib.load(SYMBOL_PCA_PATH)
    FEATURE_MEDIANS = joblib.load(FEATURE_MEDIANS_PATH)
    with open(TA_CONFIG_PATH, "r", encoding="utf-8") as f:
        TA_CONFIG = json.load(f)

    embed_model = SentenceTransformer(EMBED_MODEL, device=device)
    cls_model = joblib.load(CLS_MODEL_FULLPATH)
    reg_model = joblib.load(REG_MODEL_FULLPATH)

    cls_model_date = datetime.fromtimestamp(
        os.path.getmtime(CLS_MODEL_FULLPATH), tz=timezone.utc
    )
    reg_model_date = datetime.fromtimestamp(
        os.path.getmtime(REG_MODEL_FULLPATH), tz=timezone.utc
    )

    # jak chcesz dalej trzymać "mid" po CLS, to zostawiam tak jak miałeś
    mid_model_date = cls_model_date

    sort_predictions_file(f"data/preds/{mid_model_date.strftime('%Y%m%d_%H%M%S')}.csv")

    ta_cache = {}

    async with aiohttp.ClientSession() as session:
        client = UMFutures(binance_API_KEY, binance_SECRET_KEY)

        while True:
            # odśmiecanie failed_addresses z tych, które już są seen
            seen_set = set(processor["seen_addresses"])
            processor["failed_addresses"] = deque(
                dict.fromkeys(
                    addr
                    for addr in processor["failed_addresses"]
                    if addr not in seen_set
                ),
                maxlen=15,
            )

            new_tokens = await fetch_new_tokens(session)
            current_addresses = [t["attributes"]["address"] for t in new_tokens]
            all_addresses = list(
                set(current_addresses + list(processor["failed_addresses"]))
            )

            valid_pairs, failed_pairs = await fetch_valid_pairs_details(
                session, all_addresses, verbose=False
            )
            processor["failed_addresses"].extend(failed_pairs)

            for pair in valid_pairs:
                if pair["pairAddress"] in processor["seen_addresses"]:
                    continue

                # === FILTR CZASU: okno alertu/predykcji ===
                creation_time = datetime.fromtimestamp(
                    pair["pairCreatedAt"] / 1000, tz=timezone.utc
                )
                age = datetime.now(timezone.utc) - creation_time

                # 0..10 min (włącznie)
                if not (LAUNCH_TIME <= age <= SKIP_TIME):
                    # ważne: oznacz jako "seen", żeby nie mielić tego w kółko
                    processor["seen_addresses"].append(pair["pairAddress"])
                    continue

                try:
                    features_df = await get_features_df(
                        pair, client, TA_CONFIG, ta_cache
                    )
                    if features_df is None:
                        processor["failed_addresses"].append(pair["pairAddress"])
                        continue

                    processor["seen_addresses"].append(pair["pairAddress"])

                    input_data = await get_input(
                        df=features_df.copy(),
                        scaler=None,
                        embed_model=embed_model,
                        name_pca=NAME_PCA,
                        symbol_pca=SYMBOL_PCA,
                        name_indices=NAME_INDICES,
                        symbol_indices=SYMBOL_INDICES,
                    )
                    if input_data.isna().values.any():
                        input_data = input_data.fillna(FEATURE_MEDIANS)

                    cls_proba = cls_model.predict_proba(input_data)[0][1]
                    reg_proba = reg_model.predict(input_data)[0]

                    cls_proba_adj = cls_proba * CLS_PRECISION

                    record = {
                        "prediction_timestamp": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S.%f"
                        ),
                        "cls_prediction": f"{cls_proba*100:.2f}%",
                        "cls_prediction_adjusted": f"{cls_proba_adj*100:.2f}%",
                        "reg_prediction": f"{reg_proba:.6f}",
                        "token_name": pair["baseToken"]["name"],
                        "token_symbol": pair["baseToken"]["symbol"],
                        "url": pair["url"],
                        "cls_model_creation": cls_model_date.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "reg_model_creation": reg_model_date.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                    }

                    print(
                        f'Pair {pair["pairAddress"]} - cls: {cls_proba*100:.2f}%, reg: {reg_proba:.6f} '
                        f'name: {pair["baseToken"]["name"]}'
                    )

                    if cls_proba >= CLS_THRESHOLD_2:
                        print("##################################################")
                        print("🔔🔔🔔 POTĘŻNY TOKEN (dla jego)! 🔔🔔🔔")
                        print(
                            f"Classification: {cls_proba*100:.2f}% (adjusted {cls_proba_adj*100:.2f}%)"
                        )
                        print(
                            f"Regression: {reg_proba:.6f} (suggested {(reg_proba*cls_proba)*0.01:.6f} Sol)"
                        )
                        print(
                            f'Name: {pair["baseToken"]["name"]} Symbol: {pair["baseToken"]["symbol"]}'
                        )
                        print(f'url: {pair["url"]}')
                        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
                        print("##################################################")

                        message = (
                            f"🔔🔔🔔*POTĘŻNY TOKEN (dla jego)!*🔔🔔🔔\n\n"
                            f"*Classification:* {cls_proba*100:.2f}% (adjusted {cls_proba_adj*100:.2f}%)\n"
                            f"*Regression:* {reg_proba:.6f} (suggested {(reg_proba*cls_proba)*0.01:.6f} Sol)\n"
                            f"*Name:* {pair['baseToken']['name']}\n"
                            f"*Symbol:* {pair['baseToken']['symbol']}\n"
                            f"*URL:* {pair['url']}\n"
                            f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        await send_telegram_message(session, message, TELEGRAM_CHAT_IDs)
                        webbrowser.open(pair["url"])

                    elif cls_proba >= CLS_THRESHOLD_1:
                        print("##################################################")
                        print(
                            f"Classification: {cls_proba*100:.2f}% (adjusted {cls_proba_adj*100:.2f}%)"
                        )
                        print(
                            f"Regression: {reg_proba:.6f} (suggested {(reg_proba*cls_proba)*0.01:.6f} Sol)"
                        )
                        print(
                            f'Name: {pair["baseToken"]["name"]} Symbol: {pair["baseToken"]["symbol"]}'
                        )
                        print(f'url: {pair["url"]}')
                        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
                        print("##################################################")

                        message = (
                            f"*Potent Token*\n\n"
                            f"*Classification:* {cls_proba*100:.2f}% (adjusted {cls_proba_adj*100:.2f}%)\n"
                            f"*Regression:* {reg_proba:.6f} (suggested {(reg_proba*cls_proba)*0.01:.6f} Sol)\n"
                            f"*Name:* {pair['baseToken']['name']}\n"
                            f"*Symbol:* {pair['baseToken']['symbol']}\n"
                            f"*URL:* {pair['url']}\n"
                            f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        await send_telegram_message(session, message, TELEGRAM_CHAT_IDs)
                        webbrowser.open(pair["url"])

                    save_prediction_record(record, mid_model_date)

                except Exception as e:
                    print(f"Processing failed for {pair['pairAddress']}: {str(e)}")
                    processor["failed_addresses"].append(pair["pairAddress"])

            # proste ograniczenie wzrostu cache (żeby nie puchł bez końca)
            if len(ta_cache) > 10_000:
                ta_cache.clear()

            await asyncio.sleep(3)


if __name__ == "__main__":
    asyncio.run(main())
