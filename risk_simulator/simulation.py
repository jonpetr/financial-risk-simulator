"""Calibrate and simulate correlated geometric Brownian motion."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import eigh


@dataclass(frozen=True)
class Calibration:
    daily_log_drift: np.ndarray
    covariance: np.ndarray
    correlation: pd.DataFrame


def calibrate(prices: pd.DataFrame) -> Calibration:
    """Estimate daily log drift and sample covariance from aligned prices."""
    if len(prices) < 3 or prices.shape[1] == 0:
        raise ValueError("Calibration requires at least three common observations and one asset.")
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Calibration requires finite, positive prices.")
    log_returns = np.log(prices).diff().iloc[1:]
    return Calibration(
        daily_log_drift=log_returns.mean().to_numpy(),
        covariance=log_returns.cov().to_numpy(),
        correlation=log_returns.corr(),
    )


def simulate(
    model: Calibration,
    weights: np.ndarray,
    capital: float,
    days: int,
    num_paths: int,
    seed: int | None,
) -> np.ndarray:
    """Return portfolio values with shape (days + 1, num_paths).

    Each daily log increment is m + Lz. The eigenvalue square root
    supports singular covariance, including perfectly correlated assets.
    """
    if days < 1 or num_paths < 1 or (days + 1) * num_paths > 12_000_000:
        raise ValueError("Choose positive settings with at most 12 million path values.")
    if seed is not None and seed < 0:
        raise ValueError("Random seed must be nonnegative.")
    if (
        weights.shape != model.daily_log_drift.shape
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.isclose(weights.sum(), 1)
        or not np.isfinite(capital)
        or capital <= 0
    ):
        raise ValueError("Simulation requires normalized long-only weights and positive capital.")
    eigenvalues, eigenvectors = eigh(model.covariance)
    transform = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
    rng = np.random.default_rng(seed)
    path_values = np.empty((days + 1, num_paths))
    path_values[0] = capital
    cumulative_log_returns = np.zeros((num_paths, len(weights)))
    initial_positions = weights * capital
    try:
        with np.errstate(over="raise", invalid="raise"):
            for day in range(1, days + 1):
                shocks = rng.standard_normal(cumulative_log_returns.shape) @ transform.T
                cumulative_log_returns += model.daily_log_drift + shocks
                path_values[day] = np.exp(cumulative_log_returns) @ initial_positions
    except FloatingPointError as exc:
        raise ValueError("Simulated values overflowed. Shorten the horizon or change the calibration period.") from exc
    return path_values
