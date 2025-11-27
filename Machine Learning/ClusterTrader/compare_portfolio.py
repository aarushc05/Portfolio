#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_portfolio.py
--------------------

Compares the performance of a portfolio constructed using ml.py predictions
against simply holding the S&P 500 index.

This script:
1. Uses the existing ml.py models to make predictions
2. Constructs a portfolio based on those predictions
3. Compares the performance to the S&P 500 index
"""
from __future__ import annotations
import os
import pickle
import warnings
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Import functions from ml.py
from ml import (
    rsi, compute_features, get_sp500_tickers, load_or_download_panel,
    cluster_symbols, PRICE_COLS, N_CLUSTERS, CLUSTER_METHODS
)

# Constants
SP500_TICKER = "^GSPC"
PORTFOLIO_SIZE = 10  # Number of stocks to include in portfolio
INVESTMENT_PERIOD = 30  # Days to hold the portfolio
BACKTEST_START = datetime(2020, 1, 1)  # Start of backtest period

# Rate limiting parameters
MAX_RETRIES = 3       # Number of retries for Yahoo Finance requests
RETRY_DELAY = 60      # Base delay in seconds before retrying
JITTER = 20           # Random jitter to add to delay to avoid synchronized requests


def download_sp500_data(start_date=BACKTEST_START):
    """Download S&P 500 index data."""
    try:
        sp500_data = yf.download(
            SP500_TICKER, 
            start=start_date,
            end=datetime.today(),
            auto_adjust=False, 
            progress=False
        )
        print(f"Downloaded S&P 500 data: {len(sp500_data)} trading days")
        return sp500_data
    except Exception as e:
        print(f"Error downloading S&P 500 data: {e}")
        return pd.DataFrame()


def construct_portfolio(panel: pd.DataFrame, cluster_map: Dict[str, int], date: datetime):
    """
    Construct a portfolio of stocks based on predicted returns.
    
    Returns:
        List of tickers to include in the portfolio
    """
    # Filter data up to the given date
    hist_data = panel[panel["Date"] <= date].copy()
    
    # Group by ticker
    ticker_groups = hist_data.groupby("Ticker")
    
    predictions = []
    for ticker, group in ticker_groups:
        if len(group) < 5:  # Skip tickers with insufficient data
            continue
            
        # Get the latest data for this ticker
        latest = group.sort_values("Date").iloc[-1:]
        
        # Skip if we don't have the necessary price columns
        if any(col not in latest.columns for col in PRICE_COLS):
            continue
            
        # Get the cluster this ticker belongs to
        cluster_id = cluster_map.get(ticker, -1)
        
        # Prepare features
        X = latest[PRICE_COLS].values
        
        # Find all tickers in this cluster
        cluster_tickers = [t for t, c in cluster_map.items() if c == cluster_id]
        cluster_data = hist_data[hist_data["Ticker"].isin(cluster_tickers)]
        
        # Only proceed if we have sufficient data
        if len(cluster_data) < 30:
            continue
            
        # Train a model using cluster data
        model = LinearRegression()
        model.fit(
            cluster_data[PRICE_COLS].values,
            cluster_data["Return"].values
        )
        
        # Make prediction for this ticker
        pred_return = model.predict(X)[0]
        
        predictions.append((ticker, pred_return))
    
    # Sort by predicted return (highest first)
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Select top N stocks
    selected_tickers = [p[0] for p in predictions[:PORTFOLIO_SIZE]]
    
    return selected_tickers


def download_with_retry(tickers, start_date, end_date, retry_count=0):
    """Download data from Yahoo Finance with retry mechanism for rate limiting"""
    try:
        # Add a small delay before each request to avoid rate limiting
        time.sleep(random.uniform(1, 3))
        
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False
        )
        return data, None
    except Exception as e:
        if retry_count < MAX_RETRIES and "Rate limit" in str(e):
            # Calculate delay with exponential backoff and jitter
            delay = RETRY_DELAY * (2 ** retry_count) + random.uniform(0, JITTER)
            print(f"  Rate limit hit. Retrying in {delay:.1f} seconds (attempt {retry_count+1}/{MAX_RETRIES})")
            time.sleep(delay)
            return download_with_retry(tickers, start_date, end_date, retry_count + 1)
        else:
            return None, e

def evaluate_portfolio(tickers: List[str], start_date: datetime, end_date: datetime):
    """
    Evaluate the performance of a portfolio of stocks.
    
    Args:
        tickers: List of tickers in the portfolio
        start_date: Portfolio creation date
        end_date: Evaluation end date
        
    Returns:
        portfolio_return: Total return of the portfolio
    """
    if not tickers:
        print(f"  WARNING: No tickers selected for portfolio on {start_date}")
        return 0.0  # No tickers, no return
        
    # Download data for all tickers in a single call (more efficient)
    data, error = download_with_retry(tickers, start_date, end_date)
    
    if error:
        print(f"  Error downloading data: {error}")
        return 0.0
        
    if data is not None and not data.empty:
        print(f"  Downloaded data for {len(tickers)} tickers: {len(data)} days")
    else:
        print(f"  No data downloaded for period {start_date} to {end_date}")
        return 0.0
    
    if data.empty:
        print(f"  WARNING: No data available for any tickers in this period")
        return 0.0
    
    # Handle the case when only one ticker is downloaded (data structure is different)
    if len(tickers) == 1:
        # For single ticker, 'Adj Close' is a column, not a MultiIndex level
        if 'Adj Close' in data.columns:
            prices_df = pd.DataFrame({tickers[0]: data['Adj Close']})
        else:
            print(f"  WARNING: No Adj Close data available")
            return 0.0
    else:
        # For multiple tickers, 'Adj Close' is the first level of MultiIndex columns
        if 'Adj Close' in data.columns.levels[0]:
            prices_df = data['Adj Close']
        else:
            print(f"  WARNING: No Adj Close data available for tickers")
            return 0.0
    
    # Make sure we have data for at least one ticker
    if prices_df.empty:
        print(f"  WARNING: Empty price dataframe after processing")
        return 0.0
    
    # Handle missing data - forward fill then backward fill
    prices_df = prices_df.ffill().bfill()
    
    # Drop columns (tickers) that still have NaN values
    prices_df = prices_df.dropna(axis=1)
    
    # Make sure we still have data for at least one ticker
    if prices_df.empty or prices_df.shape[1] == 0:
        print(f"  WARNING: No valid price data after handling missing values")
        return 0.0
    
    # Make sure we have at least 2 days of data
    if len(prices_df) < 2:
        print(f"  WARNING: Insufficient data points ({len(prices_df)}) for period")
        return 0.0
    
    # Equal-weight portfolio
    prices_df['Portfolio'] = prices_df.mean(axis=1)
    
    # Calculate return
    first_value = prices_df['Portfolio'].iloc[0]
    last_value = prices_df['Portfolio'].iloc[-1]
    
    return_value = (last_value / first_value) - 1.0
    print(f"  Portfolio return: {return_value * 100:.2f}%")
    return return_value


def calculate_sp500_return(start_date: datetime, end_date: datetime):
    """Calculate the S&P 500 return for the given period."""
    # Use our retry mechanism for S&P 500 data too
    data, error = download_with_retry(SP500_TICKER, start_date, end_date)
    
    if error:
        print(f"  Error calculating S&P 500 return: {error}")
        return 0.0
    
    if data is None or data.empty:
        print(f"  WARNING: No S&P 500 data available for period {start_date} to {end_date}")
        return 0.0
        
    if len(data) < 2:
        print(f"  WARNING: Insufficient S&P 500 data points ({len(data)}) for period")
        return 0.0
        
    # Get scalar values from the Series properly to avoid deprecation warnings
    first_value = float(data["Adj Close"].iloc[0].iloc[0])
    last_value = float(data["Adj Close"].iloc[-1].iloc[0])
    
    return_value = (last_value / first_value) - 1.0
    print(f"  S&P 500 return: {return_value * 100:.2f}%")
    return return_value


def calculate_buy_and_hold_sp500(start_date):
    """
    Calculate the true buy-and-hold performance of the S&P 500 from start_date to today.
    This represents what an investor would actually experience holding the index.
    """
    print(f"Calculating buy-and-hold S&P 500 return from {start_date} to today...")
    
    try:
        # Download full S&P 500 data
        data, error = download_with_retry(SP500_TICKER, start_date, datetime.today())
        
        if error or data is None or data.empty:
            print(f"Error downloading S&P 500 data: {error}")
            return None
            
        # Extract adjusted close prices (handling MultiIndex if needed)
        try:
            if isinstance(data["Adj Close"], pd.Series):
                adj_close = data["Adj Close"]
            else:  # MultiIndex DataFrame
                adj_close = data["Adj Close"].iloc[:, 0]  # Take first column
                
            # Calculate daily returns
            daily_returns = adj_close.pct_change().dropna()
            
            # Calculate cumulative returns (compounded)
            cumulative_returns = (1 + daily_returns).cumprod() - 1
            
            # Create a dataframe for the results
            buy_hold_df = pd.DataFrame({
                "Date": daily_returns.index,
                "SP500_Return": daily_returns.values * 100,  # Convert to percent
                "SP500_Cumulative": cumulative_returns.values * 100  # Convert to percent
            })
            
            total_return = cumulative_returns.iloc[-1] * 100
            print(f"Buy-and-hold S&P 500 return: {total_return:.2f}%")
            
            # Plot for reference
            plt.figure(figsize=(10, 6))
            plt.plot(buy_hold_df["Date"], buy_hold_df["SP500_Cumulative"])
            plt.title(f"S&P 500 Buy-and-Hold Return ({start_date} to Today)")
            plt.xlabel("Date")
            plt.ylabel("Cumulative Return (%)")
            plt.grid(True)
            plt.savefig("sp500_buy_and_hold.png")
            plt.close()
            
            return buy_hold_df
            
        except Exception as e:
            print(f"Error processing S&P 500 data: {e}")
            return None
            
    except Exception as e:
        print(f"Error calculating buy-and-hold S&P 500 return: {e}")
        return None


def run_backtest(panel: pd.DataFrame, cluster_method: str, start_date: datetime):
    """
    Run a backtest comparing the portfolio strategy against buy-and-hold S&P 500.
    
    Args:
        panel: Panel data from ml.py
        cluster_method: Clustering method to use
        start_date: Start date for backtest
        
    Returns:
        backtest_results: DataFrame with performance comparison
    """
    # First calculate the true buy-and-hold S&P 500 return
    buy_hold_data = calculate_buy_and_hold_sp500(start_date)
    
    # Get the clustering map
    print(f"Clustering stocks using {cluster_method.upper()} method...")
    clusters = cluster_symbols(panel, cluster_method)
    print(f"Created {len(set(clusters.values()))} clusters with {len(clusters)} stocks")
    
    # Filter data to backtest period
    panel = panel[panel["Date"] >= start_date].copy()
    print(f"Filtered to {len(panel)} rows starting from {start_date}")
    
    # Get unique dates
    dates = sorted(panel["Date"].unique())
    print(f"Found {len(dates)} unique trading days")
    
    # Limit testing to a reasonable number of periods (up to 20 to avoid rate limiting)
    # This uses the original sampling method with non-continuous intervals
    if len(dates) > INVESTMENT_PERIOD + 20:
        step_size = (len(dates) - INVESTMENT_PERIOD) // 20
        if step_size < 1:
            step_size = 1
        test_dates = dates[:-INVESTMENT_PERIOD:step_size]
        print(f"Testing on {len(test_dates)} sample periods (every {step_size} trading days)")
    else:
        test_dates = dates[:-INVESTMENT_PERIOD]
        print(f"Testing on all {len(test_dates)} periods")
    
    results = []
    for i, date in enumerate(test_dates):
        print(f"\nBacktesting period {i+1}/{len(test_dates)}: {date}")
        
        try:
            # Construct portfolio
            portfolio_tickers = construct_portfolio(panel, clusters, date)
            
            if not portfolio_tickers:
                print(f"WARNING: No tickers selected for portfolio on {date}, skipping period")
                continue
                
            # Find appropriate end date - exactly INVESTMENT_PERIOD trading days later if possible
            date_idx = dates.index(date)
            end_idx = min(date_idx + INVESTMENT_PERIOD, len(dates) - 1)
            end_date = dates[end_idx]
            
            # Evaluate portfolio
            print(f"Evaluating portfolio performance from {date} to {end_date}:")
            portfolio_return = evaluate_portfolio(portfolio_tickers, date, end_date)
            
            # Calculate S&P 500 return
            print(f"Calculating S&P 500 performance:")
            sp500_return = calculate_sp500_return(date, end_date)
            
            # Extract buy-and-hold S&P 500 return for this period if available
            sp500_buyhold_return = 0.0
            if buy_hold_data is not None and not buy_hold_data.empty:
                # Find closest dates in the buy-and-hold data
                start_idx = buy_hold_data['Date'].searchsorted(date)
                end_idx = buy_hold_data['Date'].searchsorted(end_date)
                
                if start_idx < len(buy_hold_data) and end_idx <= len(buy_hold_data) and start_idx < end_idx:
                    # Calculate the actual S&P 500 return between these dates
                    if end_idx == len(buy_hold_data):
                        end_idx = end_idx - 1
                    start_cum = buy_hold_data['SP500_Cumulative'].iloc[start_idx]
                    end_cum = buy_hold_data['SP500_Cumulative'].iloc[end_idx]
                    sp500_buyhold_return = ((1 + end_cum/100) / (1 + start_cum/100) - 1) * 100
            
            # For each period in backtest
            results.append({
                "Date": date,
                "End_Date": end_date,
                "Portfolio_Return": portfolio_return * 100,  # Convert to percentage
                "SP500_BuyHold_Return": sp500_buyhold_return,  # Actual buy-and-hold return
                "Outperformance": (portfolio_return * 100 - sp500_buyhold_return),  # Against buy-and-hold
                "Portfolio_Tickers": portfolio_tickers
            })
            
            print(f"Period results: Portfolio: {portfolio_return*100:.2f}%, S&P 500 Buy-Hold: {sp500_buyhold_return:.2f}%, Diff: {(portfolio_return*100-sp500_buyhold_return):.2f}%")
            
        except Exception as e:
            import traceback
            print(f"Error in period {date}: {e}")
            traceback.print_exc()
    
    if not results:
        print("WARNING: No valid backtest periods found!")
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=["Date", "End_Date", "Portfolio_Return", "SP500_Return", "Outperformance", "Portfolio_Tickers"])
        
    return pd.DataFrame(results)


def calculate_true_sp500_return():
    """
    Calculate the actual S&P 500 return from BACKTEST_START to today.
    This shows the true market performance regardless of our backtesting methodology.
    """
    print(f"\nCalculating actual S&P 500 performance from {BACKTEST_START} to present...")
    
    try:
        # Download the full S&P 500 data for the period
        data, error = download_with_retry(SP500_TICKER, BACKTEST_START, datetime.today())
        
        if error or data is None or data.empty:
            print(f"  Error getting S&P 500 data: {error if error else 'No data available'}")
            return None
            
        if len(data) < 2:
            print(f"  Insufficient data points for S&P 500")
            return None
            
        # Get the first and last values - handle both MultiIndex and regular DataFrame
        try:
            if isinstance(data["Adj Close"], pd.Series):
                first_value = float(data["Adj Close"].iloc[0])
                last_value = float(data["Adj Close"].iloc[-1])
            else:  # MultiIndex DataFrame
                first_value = float(data["Adj Close"].iloc[0].iloc[0])
                last_value = float(data["Adj Close"].iloc[-1].iloc[0])
                
            # Calculate the total return
            total_return = (last_value / first_value - 1) * 100
            first_date = data.index[0] if hasattr(data.index, '__getitem__') else 'start'
            last_date = data.index[-1] if hasattr(data.index, '__getitem__') else 'end'
            
            print(f"  First S&P 500 value ({first_date}): ${first_value:.2f}")
            print(f"  Latest S&P 500 value ({last_date}): ${last_value:.2f}")
            print(f"  Actual S&P 500 total return: {total_return:.2f}%")
            
            # Plot this for visual reference
            plt.figure(figsize=(10, 6))
            plt.plot(data.index, data["Adj Close"])
            plt.title(f"S&P 500 Index ({first_date} to {last_date})")
            plt.xlabel("Date")
            plt.ylabel("Index Value ($)")
            plt.grid(True)
            plt.savefig("sp500_true_performance.png")
            plt.close()
            print(f"  True S&P 500 performance chart saved to: sp500_true_performance.png")
            
            return total_return
            
        except Exception as e:
            print(f"  Error processing S&P 500 data: {e}")
            print(f"  Data structure: {type(data.get('Adj Close', 'Unknown'))}")
            return None
            
    except Exception as e:
        print(f"  Error calculating true S&P 500 return: {e}")
        return None


def calculate_sp500_buy_and_hold(start_date: datetime, end_date: datetime):
    """Generate the S&P 500 buy-and-hold graph with the correct return"""
    plt.figure(figsize=(12, 8))
    
    # Download S&P 500 data for the entire period
    print(f"Downloading S&P 500 data from {start_date} to {end_date}...")
    sp500_data, err = download_with_retry(SP500_TICKER, start_date, end_date)
    
    if err or sp500_data is None or sp500_data.empty:
        print(f"Error getting S&P 500 data: {err if err else 'No data'}")
        return None, None, None
    
    # Calculate real S&P 500 buy-and-hold performance
    try:
        # Extract adjusted close prices (handling MultiIndex if needed)
        if isinstance(sp500_data["Adj Close"], pd.Series):
            prices = sp500_data["Adj Close"]
        else:  # MultiIndex DataFrame
            prices = sp500_data["Adj Close"].iloc[:, 0]  # Take first column
            
        # Calculate the true buy-and-hold return
        first_price = prices.iloc[0]
        sp500_normalized = prices / first_price - 1
        
        # Plot the S&P 500 line
        plt.plot(sp500_data.index, sp500_normalized * 100, 
                label="S&P 500 Buy-and-Hold", linewidth=2, color="green")
        
        # Add title and labels
        plt.title(f"S&P 500 Buy-and-Hold Return ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return (%)")
        plt.grid(True)
        
        # Get the total S&P 500 return
        sp500_total_return = sp500_normalized.iloc[-1] * 100
        print(f"S&P 500 true buy-and-hold return: {sp500_total_return:.2f}%")
        
        # Save the base S&P 500 graph for later use
        plt.savefig("sp500_buy_and_hold.png")
        
        return plt.gcf(), sp500_data, sp500_total_return
        
    except Exception as e:
        print(f"Error calculating S&P 500 performance: {e}")
        return None, None, None


def plot_performance(results: pd.DataFrame, cluster_method: str):
    """Plot the performance comparison by overlaying cluster returns onto the S&P 500 graph"""
    # Get the full date range for the backtest
    start_date = BACKTEST_START  # Use fixed start date for consistency
    end_date = datetime.today()
    
    # Check if we already have an S&P 500 base figure
    if not os.path.exists("sp500_buy_and_hold.png"):
        # Generate the S&P 500 graph to use as base
        base_fig, sp500_data, sp500_total_return = calculate_sp500_buy_and_hold(start_date, end_date)
    else:
        # Recalculate using the same dates to ensure we have the data and return value
        _, sp500_data, sp500_total_return = calculate_sp500_buy_and_hold(start_date, end_date)
        # Load the figure from the saved file
        base_fig = plt.figure(figsize=(12, 8))
        img = plt.imread("sp500_buy_and_hold.png")
        plt.imshow(img, aspect='auto')
        plt.axis('off')  # Turn off the axis
        # Create a new figure for the actual plot
        plt.figure(figsize=(12, 8))
        
        # Re-create the S&P 500 line for the legend and axis consistency
        if sp500_data is not None and len(sp500_data) > 1:
            if isinstance(sp500_data["Adj Close"], pd.Series):
                prices = sp500_data["Adj Close"]
            else:
                prices = sp500_data["Adj Close"].iloc[:, 0]
                
            first_price = prices.iloc[0]
            sp500_normalized = prices / first_price - 1
            plt.plot(sp500_data.index, sp500_normalized * 100, 
                    label="S&P 500 Buy-and-Hold", linewidth=2, color="green")
    
    # Calculate cumulative portfolio returns
    portfolio_cum = (1 + results["Portfolio_Return"] / 100).cumprod() - 1
    
    # Plot portfolio performance at each evaluation date
    plt.plot(results["End_Date"], portfolio_cum * 100, 
             label=f"{cluster_method.upper()} Portfolio", linewidth=2, color="blue", marker="o")
    
    # Add connecting lines between portfolio evaluation points
    for i in range(len(results) - 1):
        plt.plot([results["End_Date"].iloc[i], results["Date"].iloc[i+1]], 
                 [portfolio_cum.iloc[i] * 100, portfolio_cum.iloc[i] * 100], 
                 color="blue", linestyle=":", alpha=0.5)
    
    # Calculate portfolio total return
    if len(results) > 0:
        portfolio_total_return = portfolio_cum.iloc[-1] * 100
        print(f"Portfolio ({cluster_method}) total return: {portfolio_total_return:.2f}%")
    else:
        portfolio_total_return = None
    
    plt.title(f"Portfolio vs S&P 500 Performance ({cluster_method.upper()} Clustering)", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Return (%)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Calculate statistics
    port_mean = results["Portfolio_Return"].mean()
    sp500_mean = results["SP500_BuyHold_Return"].mean()
    outperf_mean = results["Outperformance"].mean()
    
    # Update title and labels
    plt.title(f"Portfolio ({cluster_method.upper()}) vs S&P 500 Performance")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (%)")
    plt.grid(True)
    plt.legend()
    
    # Add text box with statistics - simplified to only show total returns
    diff = portfolio_total_return - sp500_total_return if portfolio_total_return is not None and sp500_total_return is not None else 0
    textstr = f"""
    Total Returns: 
    {cluster_method.upper()} Portfolio: {portfolio_total_return:.1f}%
    S&P 500 Buy-and-Hold: {sp500_total_return:.1f}%
    Difference: {diff:.1f}%
    """
    
    props = dict(boxstyle="round", facecolor="white", alpha=0.8)
    plt.annotate(textstr, xy=(0.02, 0.02), xycoords="axes fraction", 
                fontsize=12, verticalalignment="bottom", bbox=props)
    
    plt.tight_layout()
    plt.savefig(f"portfolio_vs_sp500_{cluster_method}.png")
    plt.close()


def create_best_performer_graph(all_results):
    """Create a combined graph using the best performer for each period"""
    if not all_results or len(all_results) == 0:
        print("No results available to create best performer graph")
        return
    
    # Collect all dates from all methods
    all_dates = set()
    for method_results in all_results.values():
        all_dates.update(method_results["Date"].tolist())
    
    # Sort dates chronologically
    all_dates = sorted(all_dates)
    
    # Create a combined DataFrame with columns for each method
    combined_data = []
    
    # For each date, find which method performed best
    for date in all_dates:
        best_return = -float('inf')
        best_method = ""
        best_data = None
        
        # Check each method's performance on this date
        for method, results in all_results.items():
            # Find the row with this date
            date_rows = results[results["Date"] == date]
            if not date_rows.empty:
                # Get the return for this period
                period_return = date_rows["Portfolio_Return"].iloc[0]
                if period_return > best_return:
                    best_return = period_return
                    best_method = method
                    best_data = date_rows.iloc[0].to_dict()
        
        # If we found a best method for this date, add it to our data
        if best_method and best_data:
            best_data["Best_Method"] = best_method
            combined_data.append(best_data)
    
    # Convert to DataFrame
    if not combined_data:
        print("No best performer data could be generated")
        return
        
    best_performers = pd.DataFrame(combined_data)
    
    # Use the standard plot_performance function to create the graph
    print("\nCreating best performer graph (selecting best method for each period)...")
    plot_performance(best_performers, "best_performer")
    
    # Add a summary of which methods were selected
    method_counts = best_performers["Best_Method"].value_counts()
    print("\nBest performing methods selection counts:")
    for method, count in method_counts.items():
        print(f"  {method.upper()}: {count} periods ({count/len(best_performers)*100:.1f}%)")


if __name__ == "__main__":
    print("\n" + "═" * 90)
    print("PORTFOLIO PERFORMANCE VS S&P 500 COMPARISON")
    print("═" * 90)
    
    # Load panel data
    panel_data = load_or_download_panel()
    
    # Handle both tuple return format and single dataframe format for backward compatibility
    if isinstance(panel_data, tuple) and len(panel_data) == 2:
        panel_df, sp500_panel = panel_data
    else:
        panel_df = panel_data
        sp500_panel = None
        
    print(f"Loaded panel data with {len(panel_df)} rows and {len(panel_df['Ticker'].unique())} unique tickers")
    
    # First create the S&P 500 buy-and-hold baseline graph
    print("\nGenerating S&P 500 buy-and-hold baseline...")
    _, _, actual_sp500_return = calculate_sp500_buy_and_hold(BACKTEST_START, datetime.today())
    
    # Store results from all methods for later use in best performer graph
    all_results = {}
    
    for method in CLUSTER_METHODS:
        print(f"\nRunning backtest with {method.upper()} clustering...")
        
        try:
            # Run backtest with the first method
            results = run_backtest(panel_df, method, BACKTEST_START)
            
            if len(results) == 0:
                print(f"No valid results for {method.upper()} clustering. Skipping analysis.")
                continue
                
            # Save results
            results.to_csv(f"backtest_results_{method}.csv", index=False)
            
            # Plot results
            plot_performance(results, method)
            
            # Print summary
            print(f"\nResults for {method.upper()} clustering:")
            print(f"Average Portfolio Return: {results['Portfolio_Return'].mean():.2f}%")
            print(f"Average S&P 500 Buy-Hold Return: {results['SP500_BuyHold_Return'].mean():.2f}%")
            print(f"Average Outperformance: {results['Outperformance'].mean():.2f}%")
            print(f"Win Rate: {(results['Outperformance'] > 0).mean() * 100:.1f}%")
            print(f"Results saved to: backtest_results_{method}.csv")
            print(f"Performance chart saved to: portfolio_vs_sp500_{method}.png")
            
            # Store results for later use in best performer graph
            all_results[method] = results.copy()
            
            # Take a break between methods to avoid rate limiting
            if method != CLUSTER_METHODS[-1]:  # If not the last method
                wait_time = 60 + random.uniform(0, 30)  # 1-1.5 minutes
                print(f"\nTaking a break for {wait_time:.0f} seconds to avoid rate limiting...")
                time.sleep(wait_time)
        
        except Exception as e:
            import traceback
            print(f"Error running backtest for {method}: {e}")
            print("Detailed error:")
            traceback.print_exc()
    
    # Create a combined graph with the best performer for each period
    if len(all_results) > 0:
        create_best_performer_graph(all_results)
    
    print("\n" + "═" * 90)
