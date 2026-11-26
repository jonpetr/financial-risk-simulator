"""Adjusted market prices and a bundled synthetic demonstration dataset."""

import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_prices.csv"
SAMPLE_TICKERS = ("SPY", "QQQ", "IEF", "GLD")


def normalize_tickers(symbols: list[str]) -> list[str]:
    tickers = [symbol.strip().upper() for symbol in symbols]
    if any(not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-=^]{0,19}", ticker) for ticker in tickers):
        raise ValueError("Each row needs a valid ticker symbol, such as SPY or BRK-B.")
    return tickers


def align_prices(
    prices: pd.DataFrame, tickers: tuple[str, ...], start: date, end: date
) -> pd.DataFrame:
    """Keep only complete observations and preserve portfolio column order."""
    unavailable = [ticker for ticker in tickers if ticker not in prices or prices[ticker].isna().all()]
    if unavailable:
        raise ValueError(f"Price history unavailable for {', '.join(unavailable)} in the selected source.")
    prices = prices.loc[:, list(tickers)].copy()
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
    prices = prices.sort_index()
    prices = prices.loc[~prices.index.duplicated(keep="last")]
    prices = prices.loc[str(start):str(end)]
    prices = prices.replace([np.inf, -np.inf], np.nan).where(prices > 0).dropna()
    if len(prices) < 30:
        raise ValueError("The selected assets need at least 30 common price observations. Expand the date range or revise the portfolio.")
    return prices


@st.cache_data(ttl=3600, max_entries=64, show_spinner=False)
def load_prices(source: str, tickers: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    if start >= end:
        raise ValueError("The historical end date must be after the start date.")
    if source == "Offline sample":
        prices = pd.read_csv(SAMPLE_PATH, index_col="Date", parse_dates=True)
    else:
        try:
            data = yf.download(
                list(tickers), start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(), auto_adjust=True,
                group_by="column", threads=False, progress=False, timeout=15,
            )
        except Exception as exc:
            raise ValueError("Yahoo Finance could not be reached. Try again or select the offline sample.") from exc
        if data.empty or "Close" not in data:
            raise ValueError("Yahoo Finance returned no prices. Check the symbols and date range or select the offline sample.")
        prices = data["Close"]
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(tickers[0])
    return align_prices(prices, tickers, start, end)
