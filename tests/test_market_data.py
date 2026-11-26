from datetime import date

import numpy as np
import pandas as pd
import pytest

from risk_simulator.market_data import align_prices, load_prices, normalize_tickers


def test_sample_loads_offline_in_requested_order():
    prices = load_prices("Offline sample", ("GLD", "SPY"), date(2022, 1, 3), date(2022, 3, 1))
    assert prices.columns.tolist() == ["GLD", "SPY"]
    assert prices.index[-1] == pd.Timestamp("2022-03-01")
    assert prices.notna().all().all()


def test_alignment_uses_complete_positive_rows_only():
    dates = pd.bdate_range("2024-01-01", periods=35)
    prices = pd.DataFrame({"A": 100., "B": 50.}, index=dates)
    prices.loc[dates[2], "A"] = np.nan
    prices.loc[dates[4], "B"] = 0.
    aligned = align_prices(prices, ("B", "A"), dates[0].date(), dates[-1].date())
    assert len(aligned) == 33
    assert dates[2] not in aligned.index and dates[4] not in aligned.index


def test_ticker_validation():
    assert normalize_tickers([" brk-b ", "spy"]) == ["BRK-B", "SPY"]
    with pytest.raises(ValueError):
        normalize_tickers(["A B"])
