# solana-new-pool-classification

## Overview

Data + ML pipeline for discovering, enriching, and scoring newly created liquidity pools on Solana DEXs. It continuously ingests pool metadata, augments it with technical analysis (TA) signals from Binance futures, and applies trained ML models to identify potentially interesting token pairs.

The project is designed for:

- **Data collection** (new pools, metadata, price/volume/txns)
- **Feature engineering** (TA indicators, time features, text embeddings)
- **Model training** (classification + regression)
- **Real-time scoring** of new pools with alerting hooks

> **Disclaimer**: This project is for research and data analysis. It does not provide financial advice.

---

## What This Repo Contains

### Core collectors
- **`db_collector.py`** — continuously pulls new pools from GeckoTerminal, resolves details via DexScreener, and stores rows in SQLite.
- **`new_pool_sniper.py`** — older CSV-based pipeline (still useful for quick datasets).
- **`reclassify.py`** — updates/enriches historical data and produces `data/tokens_raw_reclassified.csv` for modeling.

### Scoring / alerting
- **`snipe.py`** — live scoring of new pools using trained models + TA features; can send Telegram alerts and open URLs.

### API exposure
- **`db_expose.py`** — FastAPI server to download DB or stream CSV extracts.

### Notebooks
- **`data_preprocessing.ipynb`** — end-to-end feature engineering and dataset preparation.
- **`TA_maps.ipynb`** — build TA “maps” for indicators.
- **`cls_modeling.ipynb` / `reg_modeling.ipynb`** — train classification/regression models.

---

## Setup

### Requirements
- Python 3.11.x
- CUDA-capable GPU (recommended for embedding + model inference)
- Dependencies listed in `requirements.txt`

### Installation

```bash
git clone https://github.com/philipzabicki/solanaDEXtokenCollector.git
cd solanaDEXtokenCollector
pip install -r requirements.txt
```

---

## Configuration

Create a `credentials.py` file in the project root with required keys:

```python
# Binance (TA features)
binance_API_KEY = "your_binance_key"
binance_SECRET_KEY = "your_binance_secret"

# Telegram (alerts)
TELEGRAM_TOKEN = "your_telegram_bot_token"
TELEGRAM_CHAT_IDs = ["123456789"]

# Optional: for reclassify.py (remote DB bootstrap)
URL = "https://your-host/tokens_raw.db"
INCREMENTAL_URL = "https://your-host/tokens-since"
```

---

## Usage

### 1) Collect live pool data (SQLite)

```bash
python db_collector.py
```

This creates/updates `data/tokens_raw.db` and downloads token images into `data/imgs/`.

### 2) Build/refresh modeling dataset

```bash
python reclassify.py
```

This produces `data/tokens_raw_reclassified.csv` with current metrics and metadata.

### 3) Train models (notebooks)

Run the notebooks in this order:
1. `TA_maps.ipynb`
2. `data_preprocessing.ipynb`
3. `cls_modeling.ipynb`
4. `reg_modeling.ipynb`

Outputs are stored under `models/` and `data/modeling/`.

### 4) Live scoring / pool “sniping”

Ensure the required artifacts exist in `models/`:

- `final_cls_model.joblib`
- `final_reg_model.joblib`
- `pca_name.joblib`
- `pca_symbol.joblib`
- `important_name_pca_indices.joblib`
- `important_symbol_pca_indices.joblib`
- `feature_medians.joblib`
- `selected_indicators.json`

Then run:

```bash
python snipe.py
```

The script will:
- Pull fresh pools
- Build features (TA + embeddings + time features)
- Run classification + regression models
- Emit Telegram alerts when thresholds are met

### 5) Expose the DB via API

```bash
python db_expose.py
```

Endpoints:
- `GET /download-db` — download the SQLite file
- `GET /dump` — full CSV export
- `GET /tokens` — paginated JSON
- `GET /token/{pair_address}` — single token by address

---

## Project Flow (High-Level)

1. **Collect new pools** → GeckoTerminal
2. **Resolve details** → DexScreener
3. **Store & enrich** → SQLite + missing metadata + images
4. **Reclassify** → add current metrics
5. **Engineer features** → TA + time + name/symbol embeddings
6. **Train models** → classification + regression
7. **Score live** → alert when thresholds are met

---

## Notes

- The project expects CUDA for embedding + ML steps in notebooks and `snipe.py`.
- Data collection runs continuously; consider running it in a screen/tmux session.
- The alert logic and thresholds are in `snipe.py` and can be tuned.
