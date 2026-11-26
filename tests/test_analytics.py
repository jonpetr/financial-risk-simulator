import numpy as np
import pandas as pd
import pytest

from risk_simulator.analytics import (
    historical_risk, historical_values, max_drawdown, terminal_risk, validate_portfolio,
)


def test_weight_and_amount_inputs_are_equivalent():
    symbols = ["A", "B"]
    weights, capital = validate_portfolio(symbols, [25, 75], 400)
    amount_weights, amount_capital = validate_portfolio(symbols, [100, 300], None)
    np.testing.assert_array_equal(weights, amount_weights)
    assert capital == amount_capital == 400


@pytest.mark.parametrize(
    ("symbols", "allocations", "capital"),
    [(["A", "A"], [50, 50], 100), (["A"], [90], 100),
     (["A"], [-100], None), (["A"], [np.nan], None), (["A"], [100], 0)],
)
def test_invalid_portfolio(symbols, allocations, capital):
    with pytest.raises(ValueError):
        validate_portfolio(symbols, allocations, capital)


def test_historical_buy_and_hold_allows_weights_to_drift():
    prices = pd.DataFrame({"A": [10, 20, 20], "B": [25, 25, 50]})
    values = historical_values(prices, np.array([.6, .4]), 100)
    np.testing.assert_allclose(values, [100, 160, 200])


def test_drawdown_and_annualization():
    values = pd.Series([100., 125., 100., 110.])
    assert max_drawdown(values) == pytest.approx(.2)
    returns = np.array([.25, -.2, .1])
    volatility, sharpe, drawdown = historical_risk(values, 0.)
    assert volatility == pytest.approx(returns.std(ddof=1) * np.sqrt(252))
    assert sharpe == pytest.approx(returns.mean() * 252 / volatility)
    assert drawdown == pytest.approx(.2)


def test_sharpe_is_undefined_for_zero_volatility():
    assert np.isnan(historical_risk(pd.Series([100., 100., 100.]), .02)[1])


def test_terminal_risk_fractional_tail_and_loss_probability():
    risk = terminal_risk(np.array([70, 90, 100, 110, 130]), 100, .7)
    assert risk.var == 10
    assert risk.cvar == pytest.approx((30 + .5 * 10) / 1.5)
    assert risk.expected_return == pytest.approx(0)
    assert risk.probability_of_loss == pytest.approx(.4)
    np.testing.assert_allclose(risk.percentiles, [74, 100, 126])


def test_gains_produce_negative_loss_quantiles():
    risk = terminal_risk(np.array([101, 102, 103]), 100, .95)
    assert risk.var == risk.cvar == -1
