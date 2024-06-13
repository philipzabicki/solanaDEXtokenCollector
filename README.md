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
   git clone https://github.com/philipzabicki/solanaDEXtokenCollector.git```
