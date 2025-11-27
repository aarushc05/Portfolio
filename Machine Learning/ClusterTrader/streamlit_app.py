# streamlit_app.py
"""
Ultra‑light Streamlit dashboard for the clustered‑RSI portfolio back‑test
------------------------------------------------------------------------
*Now hard‑wired to the **pre‑computed** panel file you just uploaded so the
first run finishes in seconds.*
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
import pickle

import pandas as pd
import plotly.express as px
import streamlit as st

# Import back‑test utilities  
import compare_portfolio as cp

# ────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────
DATA_FILE = Path("sp500_rsi_price_lags_max15.pkl")  # pre‑computed panel

# Update cp so that downstream calls also use this cache path
cp.CACHE_FILE = str(DATA_FILE)

# ────────────────────────────────────────────────────────────────────
# Streamlit page setup
# ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clustered‑RSI Portfolio Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Clustered‑RSI Portfolio Back‑Test (fast mode)")

st.markdown(
    r"""
    This dashboard uses a **cached S&P 500 panel** (`sp500_rsi_price_lags_max15.pkl`) so the first run
    only deserialises data instead of hitting Yahoo Finance.  From there
    all heavy results are memoised with **Streamlit cache** for
    near‑instant tweaks.
    """,
)

# ────────────────────────────────────────────────────────────────────
# Sidebar controls
# ────────────────────────────────────────────────────────────────────
sidebar = st.sidebar
sidebar.header("Parameters")

cluster_method = sidebar.selectbox(
    "Clustering algorithm",
    options=cp.CLUSTER_METHODS,
    format_func=lambda s: s.upper(),
)

portfolio_size = sidebar.slider("Portfolio size", 5, 30, cp.PORTFOLIO_SIZE)
invest_period = sidebar.slider("Investment period (trading days)", 10, 60, cp.INVESTMENT_PERIOD)

start_date = sidebar.date_input(
    "Back‑test start date", value=dt.date(2020, 1, 1), min_value=dt.date(2000, 1, 1)
)

run_btn = sidebar.button("🚀 Run back‑test")

# Apply sidebar parameters to compare_portfolio globals
cp.PORTFOLIO_SIZE = portfolio_size
cp.INVESTMENT_PERIOD = invest_period
cp.BACKTEST_START = dt.datetime.combine(start_date, dt.time())

# ────────────────────────────────────────────────────────────────────
# Helpers with Streamlit cache
# ────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading pre‑computed S&P 500 panel …")
def load_panel_from_pickle(path: Path):
    if not path.exists():
        st.error(f"Cached panel file not found: {path}")
        st.stop()
    panel_df, sp500_panel = pickle.loads(path.read_bytes())
    return panel_df  # we only need the stock panel for back‑test

@st.cache_data(show_spinner="Running back‑test …", ttl=0)
def run_test(method: str, panel_df: pd.DataFrame):
    return cp.run_backtest(panel_df, method, cp.BACKTEST_START)

# ────────────────────────────────────────────────────────────────────
# Main logic
# ────────────────────────────────────────────────────────────────────

if run_btn:
    panel_df = load_panel_from_pickle(DATA_FILE)

    with st.spinner("Clustering & simulating portfolio …"):
        results = run_test(cluster_method, panel_df)

    if results.empty:
        st.error("No valid back‑test results. Try a different parameter set.")
        st.stop()

    # ─── summary metrics ───────────────────────────────────────────
    # Calculate cumulative returns (compound growth) for both portfolio and S&P 500
    cum_portfolio = (1 + results["Portfolio_Return"] / 100).cumprod() - 1
    cum_sp500 = (1 + results["SP500_BuyHold_Return"] / 100).cumprod() - 1

    total_port_ret = cum_portfolio.iloc[-1]
    sp500_total_ret = (
        cum_sp500.iloc[-1]
        if not results["SP500_BuyHold_Return"].isna().all()
        else float("nan")
    )
    win_rate = (results["Outperformance"] > 0).mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Portfolio total return", f"{total_port_ret*100:.2f}%")
    if pd.notna(sp500_total_ret):
        col2.metric("S&P 500 buy‑and‑hold", f"{sp500_total_ret*100:.2f}%")
        diff = total_port_ret - sp500_total_ret
        col3.metric("Difference", f"{diff*100:.2f}%", delta=f"{diff*100:+.2f}%")
    else:
        col2.metric("Win rate vs S&P", f"{win_rate*100:.1f}%")
        col3.empty()

    st.divider()

    # ─── cumulative return plot ────────────────────────────────────
    plot_df = pd.DataFrame(
        {
            "End_Date": results["End_Date"].values,
            "Portfolio": cum_portfolio.values * 100,
            "S&P 500": cum_sp500.values * 100,
        }
    )
    # Melt to long format for reliable plotting
    plot_df_long = plot_df.melt(
        id_vars=["End_Date"], 
        var_name="Series", 
        value_name="Cumulative Return (%)"
    )
    fig = px.line(
        plot_df_long,
        x="End_Date",
        y="Cumulative Return (%)",
        color="Series",
        labels={"End_Date": "Date"},
        title=f"Cumulative performance — {cluster_method.upper()} portfolio vs S&P 500",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ─── expandable raw data ───────────────────────────────────────
    with st.expander("Detailed period‑by‑period results"):
        st.dataframe(results, use_container_width=True)

else:
    msg = (
        "Cached panel **loaded from disk**.  Adjust the parameters on the left and"
        " click **Run back‑test** to see results."
    )
    st.info(msg)
