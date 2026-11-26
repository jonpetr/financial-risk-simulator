import numpy as np
import pandas as pd
import pytest

from risk_simulator.simulation import calibrate, simulate


def prices_from_returns(returns):
    return pd.DataFrame(np.exp(np.vstack([np.zeros(returns.shape[1]), returns.cumsum(axis=0)])))


def test_calibration_matches_daily_log_return_moments():
    increments = np.array([[.02, .03], [-.01, .01], [.03, -.01], [-.02, -.03]])
    model = calibrate(prices_from_returns(increments))
    np.testing.assert_allclose(model.daily_log_drift, increments.mean(axis=0), atol=1e-14)
    np.testing.assert_allclose(model.covariance, np.cov(increments, rowvar=False))
    np.testing.assert_allclose(model.correlation, np.corrcoef(increments, rowvar=False))


def test_seed_reproduces_paths_and_initial_value():
    prices = pd.DataFrame({"A": [100, 102, 100, 107], "B": [200, 210, 213, 210]})
    model = calibrate(prices)
    weights = np.array([.25, .75])
    first = simulate(model, weights, 1000, 10, 100, 42)
    second = simulate(model, weights, 1000, 10, 100, 42)
    assert first.shape == (11, 100)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[0], np.full(100, 1000.))
    assert not np.array_equal(first, simulate(model, weights, 1000, 10, 100, 43))


def test_zero_covariance_preserves_distinct_buy_and_hold_growth():
    daily = np.array([.02, -.01])
    model = calibrate(prices_from_returns(np.tile(daily, (5, 1))))
    weights = np.array([.8, .2])
    paths = simulate(model, weights, 100, 10, 5, 77)
    expected = 100 * (np.exp(np.arange(11)[:, None] * daily) @ weights)
    np.testing.assert_allclose(paths, np.broadcast_to(expected[:, None], paths.shape))


def test_perfectly_correlated_assets_match_single_asset_paths():
    prices = pd.DataFrame({"A": [1., 2., 1., .5, 1.], "B": [2., 4., 2., 1., 2.]})
    model = calibrate(prices)
    paths = simulate(model, np.array([.6, .4]), 100, 1, 20_000, 5)
    log_returns = np.log(paths[-1] / 100)
    assert log_returns.var(ddof=1) == pytest.approx(model.covariance[0, 0], rel=.04)


def test_empirical_portfolio_mean_matches_gbm_expectation():
    increments = np.array([[.02, .03], [-.01, .01], [.03, -.01], [-.02, -.03]])
    model = calibrate(prices_from_returns(increments))
    weights = np.array([.4, .6])
    days = 5
    paths = simulate(model, weights, 100, days, 50_000, 99)
    expected = 100 * (weights @ np.exp(days * (model.daily_log_drift + .5 * model.covariance.diagonal())))
    assert paths[-1].mean() == pytest.approx(expected, rel=.002)


@pytest.mark.parametrize("prices", [pd.DataFrame({"A": [1., 2.]}), pd.DataFrame({"A": [1., 0., 2.]})])
def test_reject_invalid_calibration_data(prices):
    with pytest.raises(ValueError):
        calibrate(prices)
