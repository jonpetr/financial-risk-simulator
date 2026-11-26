"""Portfolio validation and historical and simulated risk measures."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def validate_portfolio(
    tickers: list[str], allocations: list[float], capital: float | None
) -> tuple[np.ndarray, float]:
    """Return initial weights and capital. None selects dollar allocations."""
    if not 1 <= len(tickers) <= 12 or len(set(tickers)) != len(tickers):
        raise ValueError("Enter 1 to 12 distinct ticker symbols.")
    values = np.asarray(allocations, dtype=float)
    if values.shape != (len(tickers),) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Each position needs a finite, positive allocation.")
    if capital is None:
        total = float(values.sum())
        if not np.isfinite(total):
            raise ValueError("Total investment is too large.")
        return values / total, total
    if not np.isfinite(capital) or capital <= 0:
        raise ValueError("Initial investment must be finite and positive.")
    if not np.isclose(values.sum(), 100, atol=0.001, rtol=0):
        raise ValueError(f"Portfolio weights must total 100%. Current total is {values.sum():.4f}%.")
    return values / values.sum(), float(capital)


def historical_values(prices: pd.DataFrame, weights: np.ndarray, capital: float) -> pd.Series:
    """Value fixed initial holdings across the common historical dates."""
    return (prices.div(prices.iloc[0]) @ weights) * capital


def max_drawdown(values: pd.Series) -> float:
    """Return the largest peak-to-trough decline as a positive fraction."""
    return float((1 - values / values.cummax()).max())


def historical_risk(values: pd.Series, risk_free_rate: float) -> tuple[float, float, float]:
    """Return annualized sample volatility, annualized Sharpe, and maximum drawdown."""
    daily = values.pct_change().dropna()
    std = float(daily.std(ddof=1))
    daily_risk_free = np.expm1(np.log1p(risk_free_rate) / TRADING_DAYS)
    sharpe = (
        float((daily.mean() - daily_risk_free) / std * np.sqrt(TRADING_DAYS))
        if std > np.finfo(float).eps else float("nan")
    )
    return std * np.sqrt(TRADING_DAYS), sharpe, max_drawdown(values)


@dataclass(frozen=True)
class TerminalRisk:
    expected_return: float
    var: float
    cvar: float
    probability_of_loss: float
    percentiles: np.ndarray


def terminal_risk(values: np.ndarray, capital: float, confidence: float) -> TerminalRisk:
    """Compute loss measures in currency and P5/P50/P95 in terminal value.

    Expected shortfall averages exactly the upper empirical tail mass, with
    fractional weight on the boundary order statistic when needed.
    """
    if not 0 < confidence < 1 or not np.isfinite(capital) or capital <= 0:
        raise ValueError("Use positive capital and a confidence level between zero and one.")
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Risk analysis requires a finite array of terminal values.")
    losses = capital - values
    tail_size = (1 - confidence) * losses.size
    ordered = np.sort(losses)[::-1]
    whole = int(tail_size)
    fractional = tail_size - whole
    tail_sum = ordered[:whole].sum()
    if fractional:
        tail_sum += fractional * ordered[whole]
    return TerminalRisk(
        expected_return=float(values.mean() / capital - 1),
        var=float(np.quantile(losses, confidence, method="inverted_cdf")),
        cvar=float(tail_sum / tail_size),
        probability_of_loss=float(np.mean(losses > 0)),
        percentiles=np.percentile(values, [5, 50, 95]),
    )
