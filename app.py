"""Streamlit entry point for the Financial Risk Simulator."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from risk_simulator.analytics import historical_risk, historical_values, terminal_risk, validate_portfolio
from risk_simulator.charts import allocation_chart, correlation_chart, history_chart, simulation_chart, terminal_chart
from risk_simulator.market_data import SAMPLE_TICKERS, load_prices, normalize_tickers
from risk_simulator.simulation import calibrate, simulate

st.set_page_config(page_title="Financial Risk Simulator", layout="wide")
st.markdown("""
<style>
[data-testid="stMainBlockContainer"] {max-width: 1550px; padding-top: 2.1rem;}
[data-testid="stSidebar"] {border-right: 1px solid #dce5f1;}
[data-testid="stSidebarContent"] {padding-top: .2rem;}
[data-testid="stMetricLabel"] p {font-size: .86rem;}
[data-testid="stMetricValue"] {font-size: clamp(1.12rem, 1.65vw, 1.62rem);}
[data-testid="stMetric"] {background: #f8fafd; padding: .75rem .85rem; border-color: #dce5f1;}
[data-testid="stHeadingWithActionElements"] h1 {font-size: clamp(1.9rem, 3.1vw, 2.7rem); letter-spacing: -.04em;}
[data-testid="stHeadingWithActionElements"] h3 {font-size: 1.05rem;}
[data-testid="stCaptionContainer"] p {color: #51647e;}
</style>
""", unsafe_allow_html=True)
st.title("Financial Risk Simulator")
st.caption("Analyze the range of portfolio outcomes under a correlated Monte Carlo model.")

with st.sidebar:
    st.header("Simulation setup")
    source = st.selectbox("Market data", ["Offline sample", "Yahoo Finance"], key="source")
    offline = source == "Offline sample"
    default_end = date(2025, 12, 31) if offline else date.today()
    default_start = date(2022, 1, 3) if offline else default_end - timedelta(days=365 * 3)
    start = st.date_input(
        "Start date", default_start, min_value=date(1900, 1, 1),
        max_value=date.today(), key=f"start-{source}", format="YYYY-MM-DD",
    )
    end = st.date_input(
        "End date", default_end, min_value=date(1900, 1, 1),
        max_value=date.today(), key=f"end-{source}", format="YYYY-MM-DD",
    )
    horizon = st.slider("Horizon · trading days", 1, 756, 252)
    count = st.select_slider("Simulation paths", [1000, 2000, 5000, 10000, 15000], value=5000)
    confidence = st.select_slider("Confidence level", [.90, .95, .975, .99], value=.95, format_func=lambda x: f"{x * 100:g}%")
    with st.expander("Model settings"):
        seed = st.number_input("Random seed", min_value=0, max_value=2**32 - 1, value=42, step=1)
        free_rate = st.number_input("Annual risk-free rate (%)", min_value=-99.0, max_value=100.0, value=2.0, step=.25)

default_weights = [45., 25., 20., 10.]
with st.expander("Portfolio allocation", expanded=True):
    controls, positions = st.columns([1, 2], gap="large")
    mode = controls.radio("Allocation method", ["Weights (%)", "Position amounts"], horizontal=True)
    by_weight = mode == "Weights (%)"
    capital_input = controls.number_input(
        "Initial investment", min_value=.01, max_value=1e12, value=100_000., step=1000.,
        format="%.2f",
    ) if by_weight else None
    controls.caption("Long-only holdings in a single currency. Add or remove rows in the table.")
    amounts = default_weights if by_weight else [value * 1000 for value in default_weights]
    with positions:
        allocation = st.data_editor(
            pd.DataFrame({"Ticker": SAMPLE_TICKERS, mode: amounts}),
            key=f"allocation-{mode}", width="stretch", hide_index=True, num_rows="dynamic",
            height=215,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", required=True),
                mode: st.column_config.NumberColumn(mode, min_value=0., required=True, format="%.4f" if by_weight else "%.2f"),
            },
            column_order=["Ticker", mode],
        )
    run = st.button("Run simulation", type="primary")

signature = (source, start, end, mode, capital_input, horizon, count, confidence, seed, free_rate,
             allocation.to_csv(index=False))
if run or (offline and "analysis" not in st.session_state):
    try:
        with st.spinner("Loading prices and simulating portfolio outcomes…"):
            tickers = normalize_tickers(allocation["Ticker"].fillna("").tolist())
            weights, capital = validate_portfolio(tickers, allocation[mode].tolist(), capital_input)
            prices = load_prices(source, tuple(tickers), start, end)
            model = calibrate(prices)
            historical = historical_values(prices, weights, capital)
            volatility, sharpe, drawdown = historical_risk(historical, free_rate / 100)
            paths = simulate(model, weights, capital, horizon, count, seed)
            risk = terminal_risk(paths[-1], capital, confidence)
            st.session_state.analysis = (
                signature, tickers, weights, capital, prices.index, historical,
                volatility, sharpe, drawdown, model.correlation, paths, risk,
            )
    except (ValueError, OSError) as error:
        st.error(str(error))

if "analysis" not in st.session_state:
    st.info("Set the allocation and run the simulation to see portfolio results.")
    st.stop()

(previous, tickers, weights, capital, dates, historical, volatility, sharpe, drawdown,
 correlation, paths, risk) = st.session_state.analysis
shown_source, _, _, _, _, shown_horizon, shown_count, shown_confidence, shown_seed, shown_rate, _ = previous
if previous != signature:
    st.info("Settings changed. The results below are from the last successful run.")

st.caption(
    f"{'Synthetic offline sample' if shown_source == 'Offline sample' else 'Yahoo Finance adjusted prices'}"
    f"  ·  {dates[0]:%d %b %Y} to {dates[-1]:%d %b %Y}  ·  {len(dates):,} common observations"
    f"  ·  {shown_horizon} trading days  ·  {shown_count:,} paths  ·  Seed {shown_seed}"
)
simulation_tab, history_tab, methodology_tab = st.tabs(
    ["Simulation results", "Historical analysis", "Methodology"]
)

with simulation_tab:
    metrics = st.columns(4, gap="small")
    metrics[0].metric("Expected horizon return", f"{risk.expected_return:+.2%}", border=True,
                      help="Arithmetic mean of simulated terminal returns.")
    metrics[1].metric(f"VaR · {shown_confidence:.1%}".replace(".0%", "%"), f"{risk.var:,.0f}", border=True,
                      help="Terminal loss quantile in the investment currency. Negative values indicate gains at the quantile.")
    metrics[2].metric(f"CVaR · {shown_confidence:.1%}".replace(".0%", "%"), f"{risk.cvar:,.0f}", border=True,
                      help="Average terminal loss in the worst tail, in the investment currency.")
    metrics[3].metric("Probability of loss", f"{risk.probability_of_loss:.2%}", border=True,
                      help="Share of simulated terminal values below the initial investment.")

    path_panel, distribution_panel = st.columns([1.28, 1], gap="medium")
    with path_panel.container(border=True):
        st.subheader("Portfolio value paths")
        st.caption("Sample paths, median, and pointwise outcome band")
        st.plotly_chart(simulation_chart(paths, shown_confidence), config={"displaylogo": False})
    with distribution_panel.container(border=True):
        st.subheader("Terminal return distribution")
        st.caption("Frequency of outcomes at the end of the horizon")
        st.plotly_chart(terminal_chart(paths[-1] / capital - 1, risk.percentiles[1] / capital - 1),
                        config={"displaylogo": False})

    scenarios, allocation_panel = st.columns([1.28, 1], gap="medium")
    with scenarios.container(border=True):
        st.subheader("Terminal scenarios")
        for column, label, value in zip(
            st.columns(3), ["Downside · P5", "Median · P50", "Upside · P95"], risk.percentiles
        ):
            column.metric(label, f"{value:,.0f}", f"{value / capital - 1:+.2%}",
                          delta_color="off", delta_arrow="off")
        st.caption("Values are in the investment currency. P5 and P95 are scenario percentiles.")
    with allocation_panel.container(border=True):
        st.subheader("Initial allocation")
        st.plotly_chart(allocation_chart(tickers, weights), config={"displaylogo": False})

with history_tab:
    metrics = st.columns(4, gap="small")
    metrics[0].metric("Historical period return", f"{historical.iloc[-1] / capital - 1:+.2%}", border=True)
    metrics[1].metric("Annualized volatility", f"{volatility:.2%}", border=True,
                      help="Sample standard deviation of daily portfolio returns × √252.")
    metrics[2].metric("Annualized Sharpe ratio", f"{sharpe:.2f}" if np.isfinite(sharpe) else "Undefined", border=True,
                      help=f"Annualized daily excess return per unit of volatility, with a {shown_rate:g}% annual risk-free rate.")
    metrics[3].metric("Maximum drawdown", f"{drawdown:.2%}", border=True,
                      help="Largest decline from an earlier peak in the historical portfolio.")
    performance, dependence = st.columns([1.28, 1], gap="medium")
    with performance.container(border=True):
        st.subheader("Historical portfolio performance")
        st.caption("Buy-and-hold backcast over the calibration period")
        st.plotly_chart(history_chart(historical), config={"displaylogo": False})
    with dependence.container(border=True):
        st.subheader("Asset correlation")
        st.caption("Pearson correlation of historical daily log returns")
        st.plotly_chart(correlation_chart(correlation), config={"displaylogo": False})

with methodology_tab:
    st.markdown("""
### Model and assumptions

The simulator estimates each asset’s average daily log return and the sample covariance of
daily log returns on common observation dates. Each trading day it draws a multivariate
normal log return from those estimates and compounds individual holdings. The initial
allocation is held throughout each path and weights drift as prices change.

### Reading the results

- **Value at Risk (VaR)** is the selected quantile of terminal losses, measured from the
  initial investment. **Conditional VaR (CVaR)** averages the worst tail of those losses.
- The **outcome band** shows central percentiles at each trading day. It is a pointwise
  model interval and does not describe the chance that an entire path stays inside it.
- **Annualized volatility, Sharpe ratio, and maximum drawdown** summarize the
  historical buy-and-hold backcast. With the offline sample, these describe synthetic observations.
- **P5, P50 and P95** describe the downside, median, and upside terminal scenarios.
  Outcomes beyond P5 and P95 remain possible.

### Limitations

Simulations do not predict future returns. Gaussian log returns and constant drift,
volatility and correlation can miss jumps, heavy tails and changing market regimes.
Historical drift is especially uncertain. The model assumes a shared currency and
omits FX conversion, taxes, trading costs and liquidity constraints. The offline
sample contains synthetic prices and is available without a network connection.
""")
