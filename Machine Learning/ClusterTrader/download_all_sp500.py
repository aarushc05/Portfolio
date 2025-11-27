#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_all_sp500.py
---------------------
Downloads data for all S&P 500 stocks in batches to avoid rate limiting,
processes the data, and saves it to a cache file for future use.
"""
import os
import pickle
import time
import random
from datetime import datetime
from typing import List, Tuple

import pandas as pd
import yfinance as yf
import numpy as np

# Configuration - matching the settings in ml.py
START_DATE = datetime(2015, 1, 1)
END_DATE = datetime.today()
MAX_LAGS = 15
CACHE_FILE = f"sp500_rsi_price_lags_max{MAX_LAGS}.pkl"  # Same file as used in ml.py
SP500_INDEX = "^GSPC"

# Batch processing config
BATCH_SIZE = 20  # Number of tickers to download at once
BATCH_DELAY = 60  # Seconds to wait between batches
RETRY_ATTEMPTS = 3  # Number of retries for failed downloads


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index for a price series"""
    diff = series.diff()
    up, down = diff.clip(lower=0), -diff.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - 100 / (1 + rs)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RSI, price lags, and return for a dataframe"""
    df = df.copy()
    df["RSI"] = rsi(df["Adj Close"])
    for lag in range(MAX_LAGS):
        df[f"P{lag}"] = df["Adj Close"].shift(lag)
    df["Return"] = df["Adj Close"].pct_change().shift(-1)
    return df.dropna()


def get_sp500_tickers() -> List[str]:
    """Get all S&P 500 tickers from Wikipedia"""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    return pd.read_html(url)[0]["Symbol"].tolist()


def download_with_retry(ticker, retry_count=0):
    """Download data for a single ticker with retry logic"""
    try:
        # Add small random delay to avoid hitting rate limits
        time.sleep(random.uniform(0.5, 2))
        
        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=False,
            progress=False
        )
        
        if df.empty:
            print(f"No data found for {ticker}")
            return None
            
        return df
    except Exception as e:
        if retry_count < RETRY_ATTEMPTS and "Rate limit" in str(e):
            delay = (2 ** retry_count) * 10 + random.uniform(0, 5)
            print(f"Rate limit hit for {ticker}. Retrying in {delay:.1f} seconds...")
            time.sleep(delay)
            return download_with_retry(ticker, retry_count + 1)
        else:
            print(f"Failed to download {ticker}: {e}")
            return None


def download_all_sp500_data(force_download=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download and process data for all S&P 500 stocks in batches"""
    # Check for existing cache file
    if os.path.exists(CACHE_FILE) and not force_download:
        print(f"Loading cached data from {CACHE_FILE}")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
            
    if os.path.exists(CACHE_FILE) and force_download:
        print(f"Cache file {CACHE_FILE} exists, but force_download=True, downloading fresh data...")
    
    # Get all S&P 500 tickers
    all_tickers = get_sp500_tickers()
    print(f"Found {len(all_tickers)} S&P 500 tickers")
    
    # Download S&P 500 index data first
    print(f"Downloading S&P 500 index data...")
    sp500_df = download_with_retry(SP500_INDEX)
    
    if sp500_df is not None and not sp500_df.empty:
        sp500_df = compute_features(sp500_df)
        sp500_df["Ticker"] = SP500_INDEX
        sp500_df = sp500_df.reset_index()
        sp500_panel = sp500_df[
            ["Date", "Ticker", "RSI"] + [f"P{lag}" for lag in range(MAX_LAGS)] + ["Return"]
        ]
        print(f"Successfully processed S&P 500 index data: {len(sp500_panel)} rows")
    else:
        print("WARNING: Could not download S&P 500 index data")
        sp500_panel = pd.DataFrame()
    
    # Process stock tickers in batches
    panel_data = []
    total_batches = (len(all_tickers) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_num, batch_start in enumerate(range(0, len(all_tickers), BATCH_SIZE)):
        batch_tickers = all_tickers[batch_start:batch_start + BATCH_SIZE]
        print(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch_tickers)} tickers)")
        
        successful_tickers = 0
        for ticker in batch_tickers:
            try:
                # Replace problematic ticker symbols
                if "." in ticker:
                    ticker = ticker.replace(".", "-")
                
                df = download_with_retry(ticker)
                if df is None or df.empty:
                    continue
                
                df = compute_features(df)
                df["Ticker"] = ticker
                df = df.reset_index()
                
                panel_data.append(
                    df[["Date", "Ticker", "RSI"] + [f"P{lag}" for lag in range(MAX_LAGS)] + ["Return"]]
                )
                successful_tickers += 1
                print(f"  Successfully processed {ticker}")
                
            except Exception as e:
                print(f"  Error processing {ticker}: {e}")
        
        print(f"Batch {batch_num + 1} complete: {successful_tickers}/{len(batch_tickers)} successful")
        
        # Add delay between batches except for the last batch
        if batch_num < total_batches - 1:
            delay = BATCH_DELAY + random.uniform(-10, 10)
            print(f"Waiting {delay:.1f} seconds before next batch...")
            time.sleep(delay)
    
    # Combine all processed data
    if panel_data:
        panel_df = pd.concat(panel_data, ignore_index=True)
        print(f"Final panel data: {len(panel_df)} rows, {panel_df['Ticker'].nunique()} tickers")
        
        # Save to cache file
        with open(CACHE_FILE, "wb") as f:
            pickle.dump((panel_df, sp500_panel), f)
        print(f"Saved data to {CACHE_FILE}")
        
        return panel_df, sp500_panel
    else:
        print("ERROR: Failed to download any stock data")
        return pd.DataFrame(), sp500_panel


if __name__ == "__main__":
    print("Starting S&P 500 data download...")
    start_time = time.time()
    
    # Force download new data even if cache exists
    force_download = True
    
    panel_df, sp500_panel = download_all_sp500_data(force_download)
    
    elapsed = time.time() - start_time
    print(f"Download completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Total stocks downloaded: {panel_df['Ticker'].nunique()}")
    print(f"Total data points: {len(panel_df)}")
    print(f"Date range: {panel_df['Date'].min()} to {panel_df['Date'].max()}")
    print(f"S&P 500 index data points: {len(sp500_panel)}")
    
    # Print instructions for next steps
    print("\nNext steps:")
    print("1. The data has been saved to the same cache file that ml.py uses")
    print("2. You can now run ml.py or compare_portfolio.py directly")
    print("3. The scripts will automatically use the full S&P 500 data")
    print("\nNote: You may need to modify the LIMIT_TICKERS parameter in ml.py to None")
    print("to use all the downloaded stocks instead of just the first 40.")
