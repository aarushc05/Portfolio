# ClusterTrader

A quantitative trading system that uses machine learning clustering to construct stock portfolios and backtest their performance against the S&P 500.

## Overview

ClusterTrader clusters S&P 500 stocks by their RSI (Relative Strength Index) patterns, trains separate return prediction models per cluster, and selects top performers for a portfolio. The hypothesis is that cluster-specific models can better capture stock behavior than a single global model.

## How It Works

1. **Feature Engineering**: Compute RSI (14-day) and 15 price lags for each stock
2. **Clustering**: Group stocks by RSI patterns using PCA dimensionality reduction followed by clustering (K-Means, GMM, or Hierarchical)
3. **Prediction**: Train Linear Regression models per cluster to predict next-day returns
4. **Portfolio Construction**: Select top N stocks with highest predicted returns
5. **Backtesting**: Hold portfolio for a fixed period, compare returns to S&P 500 buy-and-hold

## Files

| File | Description |
|------|-------------|
| `ml.py` | Core ML pipeline: RSI calculation, clustering, walk-forward cross-validation |
| `compare_portfolio.py` | Portfolio backtesting engine with S&P 500 comparison |
| `download_all_sp500.py` | Batch data downloader with rate limiting for Yahoo Finance |
| `generate_best_performer.py` | Creates oracle portfolio using best method per period |
| `streamlit_app.py` | Interactive dashboard for visualizing backtest results |

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install pandas numpy yfinance scikit-learn matplotlib plotly streamlit scipy lxml html5lib

# Download data (takes 30-60 minutes due to rate limiting)
python download_all_sp500.py

# Run dashboard
python -m streamlit run streamlit_app.py
```

## Usage

### Dashboard

The Streamlit dashboard allows you to:
- Select clustering algorithm (K-Means, GMM, Hierarchical)
- Adjust portfolio size (5-30 stocks)
- Set investment period (10-60 trading days)
- Choose backtest start date

### Command Line

```bash
# Run ML evaluation (compares global vs cluster models)
python ml.py

# Run full portfolio backtest
python compare_portfolio.py

# Generate best performer analysis from existing results
python generate_best_performer.py
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PORTFOLIO_SIZE` | 10 | Number of stocks in portfolio |
| `INVESTMENT_PERIOD` | 30 | Trading days to hold portfolio |
| `N_CLUSTERS` | 2 | Number of clusters |
| `MAX_LAGS` | 15 | Price lag features (P0-P14) |

## Output

- `backtest_results_*.csv`: Period-by-period returns for each clustering method
- `portfolio_vs_sp500_*.png`: Performance comparison charts
- `sp500_rsi_price_lags_max15.pkl`: Cached data panel


