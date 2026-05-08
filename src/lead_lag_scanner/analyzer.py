"""Lead-lag analysis.

Given a tidy trades DataFrame (see :mod:`lead_lag_scanner.storage`), for every
(symbol, exchange-pair) combination we:

1. Resample last-trade prices on a uniform grid.
2. Compute log-returns on each side.
3. Compute Pearson correlation across a configurable lag grid.
4. Locate the lag that maximises absolute correlation; the leader is the side
   whose returns ``lead`` the other.
5. Estimate a confidence interval on the optimal lag via block-bootstrap.

The analyser is deterministic: callers wanting reproducibility should pass a
``rng`` (a ``numpy.random.Generator``) into :func:`bootstrap_lag_ci`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from .config import AnalyzerConfig, BootstrapConfig, LagGrid


@dataclass(frozen=True, slots=True)
class LeadLagResult:
    symbol: str
    exchange_a: str
    exchange_b: str
    n_obs: int
    best_lag_seconds: float
    best_correlation: float
    correlation_at_zero: float
    leader: str  # exchange_a, exchange_b, or "none"
    follower: str  # the other side, or "none"
    lag_ci_low_seconds: float
    lag_ci_high_seconds: float
    follower_return_std: float


def _resample_last_price(df: pd.DataFrame, resample_seconds: float) -> pd.Series:
    """Return last-trade price on a uniform 1-second (or other) grid.

    ``df`` must contain ``timestamp_ms`` (int64) and ``price`` (float).
    Empty input ⇒ empty Series with a DatetimeIndex.
    """

    if df.empty:
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC"))
    s = pd.Series(
        df["price"].to_numpy(dtype="float64"),
        index=pd.to_datetime(df["timestamp_ms"].to_numpy(), unit="ms", utc=True),
        name="price",
    )
    s = s.sort_index()
    rule = f"{int(resample_seconds * 1000)}ms"
    return s.resample(rule).last().ffill().dropna()


def _aligned_returns(
    a: pd.Series,
    b: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute log-returns on the intersection of two price series."""

    if a.empty or b.empty:
        return np.array([]), np.array([])
    common_index = a.index.intersection(b.index)
    if len(common_index) < 3:
        return np.array([]), np.array([])
    a_aligned = a.reindex(common_index)
    b_aligned = b.reindex(common_index)
    ra = np.diff(np.log(a_aligned.to_numpy(dtype="float64")))
    rb = np.diff(np.log(b_aligned.to_numpy(dtype="float64")))
    return ra, rb


def _lag_grid_array(grid: LagGrid, resample_seconds: float) -> np.ndarray:
    """Return the lag grid in *integer steps of resample_seconds*.

    A lag of 0.5s with resample_seconds=0.1s ⇒ +5 steps.
    Non-multiples of resample_seconds are rounded to the nearest integer step.
    """

    raw = np.arange(grid.start, grid.stop + grid.step / 2, grid.step)
    steps = np.round(raw / resample_seconds).astype(int)
    return np.unique(steps)


def cross_correlation(
    ra: np.ndarray,
    rb: np.ndarray,
    lag_steps: np.ndarray,
) -> np.ndarray:
    """Compute Pearson correlation between ``ra`` and ``rb`` shifted by ``lag``.

    Positive lag ⇒ ``rb`` is shifted forward in time relative to ``ra``,
    i.e. ``corr[i] = corr(ra[t], rb[t + lag_steps[i]])``.

    Returns an array of correlations of the same length as ``lag_steps``.
    NaN is returned when the overlap shrinks below 3 points or variance
    collapses to zero.
    """

    if ra.size == 0 or rb.size == 0:
        return np.full(lag_steps.shape, np.nan)
    out = np.empty(lag_steps.shape, dtype="float64")
    n = min(ra.size, rb.size)
    for i, lag in enumerate(lag_steps):
        if lag >= 0:
            x = ra[: n - lag]
            y = rb[lag : lag + (n - lag)]
        else:
            k = -lag
            x = ra[k : k + (n - k)]
            y = rb[: n - k]
        if x.size < 3:
            out[i] = np.nan
            continue
        sx = x.std(ddof=1)
        sy = y.std(ddof=1)
        if sx == 0 or sy == 0 or not np.isfinite(sx) or not np.isfinite(sy):
            out[i] = np.nan
            continue
        out[i] = float(np.corrcoef(x, y)[0, 1])
    return out


def bootstrap_lag_ci(
    ra: np.ndarray,
    rb: np.ndarray,
    lag_steps: np.ndarray,
    config: BootstrapConfig,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Block-bootstrap a 95% CI on the argmax-|corr| lag (in *steps*).

    Returns ``(low_steps, high_steps)``.
    """

    if rng is None:
        rng = np.random.default_rng(0)
    n = min(ra.size, rb.size)
    if n < config.block_size * 3:
        return float("nan"), float("nan")
    block_starts = np.arange(0, n - config.block_size + 1)
    n_blocks = max(1, n // config.block_size)
    best_lags: list[int] = []
    for _ in range(config.n_iter):
        chosen = rng.choice(block_starts, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, s + config.block_size) for s in chosen])
        idx = idx[idx < n]
        sample_corrs = cross_correlation(ra[idx], rb[idx], lag_steps)
        if np.all(np.isnan(sample_corrs)):
            continue
        best_lags.append(int(lag_steps[np.nanargmax(np.abs(sample_corrs))]))
    if not best_lags:
        return float("nan"), float("nan")
    arr = np.array(best_lags, dtype="float64")
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def analyze_pair(
    symbol: str,
    exchange_a: str,
    exchange_b: str,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    config: AnalyzerConfig,
    *,
    rng: np.random.Generator | None = None,
) -> LeadLagResult | None:
    """Analyse a single ``(symbol, exchange_a, exchange_b)`` triple.

    Returns ``None`` if the pair has fewer than ``config.min_obs`` overlapping
    observations or returns are degenerate.
    """

    pa = _resample_last_price(df_a, config.resample_seconds)
    pb = _resample_last_price(df_b, config.resample_seconds)
    ra, rb = _aligned_returns(pa, pb)
    n_obs = min(ra.size, rb.size)
    if n_obs < config.min_obs:
        return None

    lag_steps = _lag_grid_array(config.lag_grid, config.resample_seconds)
    corrs = cross_correlation(ra, rb, lag_steps)
    if np.all(np.isnan(corrs)):
        return None

    best_idx = int(np.nanargmax(np.abs(corrs)))
    best_lag_steps = int(lag_steps[best_idx])
    best_lag_seconds = best_lag_steps * config.resample_seconds
    best_corr = float(corrs[best_idx])

    zero_idx_arr = np.where(lag_steps == 0)[0]
    corr_at_zero = float(corrs[int(zero_idx_arr[0])]) if zero_idx_arr.size > 0 else float("nan")

    if best_lag_steps > 0:
        leader, follower = exchange_a, exchange_b
    elif best_lag_steps < 0:
        leader, follower = exchange_b, exchange_a
    else:
        leader, follower = "none", "none"

    ci_low_steps, ci_high_steps = bootstrap_lag_ci(ra, rb, lag_steps, config.bootstrap, rng=rng)
    ci_low_seconds = ci_low_steps * config.resample_seconds
    ci_high_seconds = ci_high_steps * config.resample_seconds

    follower_returns = rb if follower == exchange_b else ra
    follower_std = float(follower_returns.std(ddof=1)) if follower_returns.size > 1 else 0.0

    return LeadLagResult(
        symbol=symbol,
        exchange_a=exchange_a,
        exchange_b=exchange_b,
        n_obs=n_obs,
        best_lag_seconds=best_lag_seconds,
        best_correlation=best_corr,
        correlation_at_zero=corr_at_zero,
        leader=leader,
        follower=follower,
        lag_ci_low_seconds=ci_low_seconds,
        lag_ci_high_seconds=ci_high_seconds,
        follower_return_std=follower_std,
    )


def analyze_all(
    trades: pd.DataFrame,
    config: AnalyzerConfig,
    *,
    rng: np.random.Generator | None = None,
) -> list[LeadLagResult]:
    """Run :func:`analyze_pair` over every (symbol, exchange-pair) combination
    present in ``trades``.
    """

    if trades.empty:
        return []

    results: list[LeadLagResult] = []
    by_symbol: dict[str, dict[str, pd.DataFrame]] = {}
    for symbol_val, sym_group in trades.groupby("symbol", sort=True):
        symbol = str(symbol_val)
        by_symbol[symbol] = {}
        for exchange_val, ex_group in sym_group.groupby("exchange", sort=True):
            by_symbol[symbol][str(exchange_val)] = ex_group

    for symbol, by_exchange in by_symbol.items():
        exchanges = sorted(by_exchange.keys())
        for ex_a, ex_b in combinations(exchanges, 2):
            result = analyze_pair(
                symbol,
                ex_a,
                ex_b,
                by_exchange[ex_a],
                by_exchange[ex_b],
                config,
                rng=rng,
            )
            if result is not None:
                results.append(result)
    return results
