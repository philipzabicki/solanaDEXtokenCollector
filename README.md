# Solana DEX Token Collector

## Overview

Solana DEX Token Collector is a collection of Python scripts designed to scrape and fetch data from decentralized exchanges (DEXs) on the Solana blockchain.
The primary focus is on identifying and analyzing new liquidity pools for potential high-risk investing opportunities.
It leverages machine learning models to predict the potential of new token pairs and offers tools for comprehensive data analysis.

## Features

- **Token Classification:** Classify tokens based on various metrics and predictive models.
- **Name Collection:** Gather token names and symbols from different Solana DEXs to be used with FastText models for word to vec embedding.
- **New Pool Detection:** Detect and snipe new liquidity pools to identify potential investment opportunities.
- **Machine Learning Models:** Use trained models to predict the viability of new token pairs.

## Requirements

- Python 3.11.x
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/philipzabicki/solanaDEXtokenCollector.git

2. Navigate to the project directory:
   ```bash
   cd solanaDEXtokenCollector

3. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt

## Usage
### Sniping New Pools
The snipe.py script is designed to detect and analyze new liquidity pools. Here’s how to use it:

1. Ensure you have the necessary trained models in the models directory:
- 'token_names_model.bin'
- 'token_symbols_model.bin'
- 'scaler_model.pkl'
- 'rf_model.joblib'

2. Run the script:
   ```bash
   python snipe.py
