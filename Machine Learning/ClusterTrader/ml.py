#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cluster_price_lag_lr_fixed_cfg.py
---------------------------------

Runs the fixed best‑found configuration on the **first 40** S&P‑500
symbols and compares results to actual S&P 500 index performance.

Fixed hyper‑parameters
• 40 tickers
• 2 clusters
• 3 walk‑forward CV splits
• window size = 2 (features P0, P1)
"""
from __future__ import annotations
import os, pickle, warnings
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import ttest_rel

# ───────────── fixed parameters ─────────────
LIMIT_TICKERS   = None               # None means use all S&P 500 stocks
START_DATE      = datetime(2015, 1, 1)
MAX_LAGS        = 15
REG_WIN         = 2                  # P0, P1
N_CLUSTERS      = 2
N_SPLITS        = 3
CLUSTER_METHODS = ["kmeans", "gmm", "hier"]
CACHE_FILE      = f"sp500_rsi_price_lags_max{MAX_LAGS}.pkl"
SP500_INDEX     = "^GSPC"            # S&P 500 index ticker
# ────────────────────────────────────────────


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    diff = series.diff()
    up, down = diff.clip(lower=0), -diff.clip(upper=0)
    ma_up   = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - 100 / (1 + rs)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["RSI"] = rsi(df["Adj Close"])
    for lag in range(MAX_LAGS):
        df[f"P{lag}"] = df["Adj Close"].shift(lag)
    df["Return"] = df["Adj Close"].pct_change().shift(-1)
    return df.dropna()


def get_sp500_tickers(n: int | None = None) -> List[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    return pd.read_html(url)[0]["Symbol"].tolist()[:n]


def load_or_download_panel() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if os.path.exists(CACHE_FILE):
        print(f"Loaded cached panel → {CACHE_FILE}")
        cache_data = pickle.load(open(CACHE_FILE, "rb"))
        if isinstance(cache_data, tuple) and len(cache_data) == 2:
            return cache_data
        else:
            # For backward compatibility with old cache format
            print("Old cache format detected, redownloading data...")
    
    # Download S&P 500 index data
    try:
        sp500_df = yf.download(SP500_INDEX, start=START_DATE, end=datetime.today(),
                            auto_adjust=False, progress=False)
        if not sp500_df.empty:
            sp500_df = compute_features(sp500_df)
            sp500_df["Ticker"] = SP500_INDEX
            sp500_df = sp500_df[
                ["Ticker", "RSI"] + [f"P{lag}" for lag in range(MAX_LAGS)] + ["Return"]
            ]
        else:
            warnings.warn(f"Failed to download {SP500_INDEX} data")
            sp500_df = pd.DataFrame()  # Empty DataFrame as fallback
    except Exception as exc:
        warnings.warn(f"Error downloading {SP500_INDEX}: {exc}")
        sp500_df = pd.DataFrame()  # Empty DataFrame as fallback
        
    # Download individual stock data
    tickers = get_sp500_tickers(LIMIT_TICKERS)
    panel = []
    for tkr in tickers:
        try:
            df = yf.download(tkr, start=START_DATE, end=datetime.today(),
                             auto_adjust=False, progress=False)
            if df.empty:
                continue
            df = compute_features(df)
            df["Ticker"] = tkr
            panel.append(
                df[
                    ["Ticker", "RSI"]
                    + [f"P{lag}" for lag in range(MAX_LAGS)]
                    + ["Return"]
                ]
            )
        except Exception as exc:
            warnings.warn(f"Skip {tkr}: {exc}")

    if not panel:
        raise RuntimeError("All downloads failed — check network or ticker list.")

    panel_df = (
        pd.concat(panel)
        .reset_index()
        .rename(columns={"Date": "Date"})
    )
    
    # Prepare S&P 500 index dataframe
    if not sp500_df.empty:
        sp500_panel = sp500_df.reset_index().rename(columns={"Date": "Date"})
    else:
        sp500_panel = pd.DataFrame()  # Empty DataFrame as fallback
    
    # Save both dataframes to cache
    cache_data = (panel_df, sp500_panel)
    pickle.dump(cache_data, open(CACHE_FILE, "wb"))
    print(f"Saved panel → {CACHE_FILE}  (shape {panel_df.shape})")
    return cache_data


def cluster_symbols(panel: pd.DataFrame, method: str) -> Dict[str, int]:
    feats, syms = [], []
    for sym, grp in panel.groupby("Ticker"):
        feats.append(grp.sort_values("Date")["RSI"].values)
        syms.append(sym)

    max_len = max(map(len, feats))
    padded = np.vstack(
        [np.pad(f, (max_len - len(f), 0), constant_values=np.nan) for f in feats]
    )
    padded = pd.DataFrame(padded).ffill(axis=1).bfill(axis=1).values
    emb = PCA(n_components=1, random_state=0).fit_transform(padded)

    if method == "kmeans":
        labels = KMeans(n_clusters=N_CLUSTERS, n_init="auto",
                        random_state=0).fit_predict(emb)
    elif method == "gmm":
        labels = GaussianMixture(n_components=N_CLUSTERS,
                                 random_state=0).fit_predict(emb)
    elif method == "hier":
        labels = AgglomerativeClustering(n_clusters=N_CLUSTERS,
                                         linkage="ward").fit_predict(emb)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    return dict(zip(syms, labels))


PRICE_COLS = [f"P{lag}" for lag in range(REG_WIN)]  # P0, P1


def walk_forward_eval(
    panel: pd.DataFrame, cluster_map: Dict[str, int], sp500_panel: pd.DataFrame = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    g_arr, c_arr, sp_arr = [], [], []

    panel = panel.sort_values("Date").reset_index(drop=True)
    X_all = panel[PRICE_COLS].values
    y_all = panel["Return"].values
    
    # Prepare S&P 500 data if available
    has_sp500_data = sp500_panel is not None and not sp500_panel.empty
    if has_sp500_data:
        sp500_panel = sp500_panel.sort_values("Date")
        sp500_dates = set(sp500_panel["Date"])

    for train_idx, test_idx in tscv.split(X_all):
        X_tr, y_tr = X_all[train_idx], y_all[train_idx]
        X_te, y_te = X_all[test_idx], y_all[test_idx]

        g_lr    = LinearRegression().fit(X_tr, y_tr)
        y_hat_g = g_lr.predict(X_te)
        g_arr.append(np.sqrt(mean_squared_error(y_te, y_hat_g)))

        train_df = panel.iloc[train_idx].copy()
        test_df  = panel.iloc[test_idx].copy()

        train_df["cid"] = train_df["Ticker"].map(cluster_map).fillna(-1).astype(int)
        test_df["cid"]  = test_df["Ticker"].map(cluster_map).fillna(-1).astype(int)

        y_hat_c = pd.Series(y_hat_g, index=test_idx, dtype=float)

        for cid, sub_train in train_df.groupby("cid", dropna=False):
            sub_test = test_df[test_df["cid"] == cid]
            if sub_test.empty:
                continue
            lr_c = LinearRegression().fit(
                sub_train[PRICE_COLS].values, sub_train["Return"].values
            )
            y_hat_c.loc[sub_test.index] = lr_c.predict(sub_test[PRICE_COLS].values)

        c_arr.append(np.sqrt(mean_squared_error(y_te, y_hat_c.values)))
        
        # Compare with S&P 500 performance if data is available
        if has_sp500_data:
            # Find matching dates in test set
            test_dates = set(test_df["Date"])
            common_dates = test_dates.intersection(sp500_dates)
            
            if common_dates:
                # Get S&P 500 data for common dates
                sp500_test = sp500_panel[sp500_panel["Date"].isin(common_dates)]
                
                # Use same model type for fair comparison
                sp500_train = sp500_panel[sp500_panel["Date"].isin(train_df["Date"])]
                
                if not sp500_train.empty and not sp500_test.empty:
                    sp_lr = LinearRegression().fit(
                        sp500_train[PRICE_COLS].values, sp500_train["Return"].values
                    )
                    sp_pred = sp_lr.predict(sp500_test[PRICE_COLS].values)
                    sp_rmse = np.sqrt(mean_squared_error(sp500_test["Return"].values, sp_pred))
                    sp_arr.append(sp_rmse)
                else:
                    # Fallback if we don't have matching data
                    sp_arr.append(np.nan)
            else:
                sp_arr.append(np.nan)
        else:
            sp_arr.append(np.nan)

    return np.array(g_arr), np.array(c_arr), np.array(sp_arr)


if __name__ == "__main__":
    panel, sp500_panel = load_or_download_panel()
    print("\n" + "═" * 100)
    
    # Print header
    print(f"{'METHOD':^11} | {'GLOBAL RMSE':^15} | {'CLUSTER RMSE':^15} | {'IMPROVEMENT':^15} | {'P-VALUE':^8} | {'S&P 500 RMSE':^15} | {'VS S&P':^15}")
    print("─" * 100)

    for method in CLUSTER_METHODS:
        clusters = cluster_symbols(panel, method)
        g_rmses, c_rmses, sp_rmses = walk_forward_eval(panel, clusters, sp500_panel)

        g_mean, c_mean = g_rmses.mean(), c_rmses.mean()
        improvement = g_mean - c_mean
        _, p_val = ttest_rel(g_rmses, c_rmses)
        
        # S&P 500 comparison
        valid_sp_rmses = sp_rmses[~np.isnan(sp_rmses)]
        if len(valid_sp_rmses) > 0:
            sp_mean = valid_sp_rmses.mean()
            c_vs_sp = c_mean - sp_mean
            sp_info = f"{sp_mean:.6e} | {c_vs_sp:.2e}"
        else:
            sp_info = "N/A | N/A"

        folds = ", ".join(f"{g:.2e}/{c:.2e}" for g, c in zip(g_rmses, c_rmses))

        print(f"{method.upper():11} | "
              f"{g_mean:.6e} | "
              f"{c_mean:.6e} | "
              f"{improvement:.2e} | "
              f"{p_val:.4f} | "
              f"{sp_info}")
        
        # Print fold details in a separate line with indentation
        print(f"{'':11}   Folds (global/cluster): {folds}")

    print("\n" + "═" * 100)
    print("INTERPRETATION:")
    print("- GLOBAL RMSE: Error when predicting returns with one model for all stocks")
    print("- CLUSTER RMSE: Error when predicting returns with separate models for each cluster")
    print("- IMPROVEMENT: Reduction in error from clustering (positive is better)")
    print("- P-VALUE: Statistical significance of the improvement")
    print("- S&P 500 RMSE: Error when predicting S&P 500 index returns")
    print("- VS S&P: Difference between Cluster RMSE and S&P 500 RMSE (negative means our model is better)")
    print("═" * 100)
