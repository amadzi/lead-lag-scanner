"""Lead-lag analysis with cross-exchange clock-skew calibration and
active-bar masking.

Given a tidy trades DataFrame (see :mod:`lead_lag_scanner.storage`), for every
(symbol, exchange-pair) combination we:

1. **Calibrate clocks across exchanges.** For each exchange we compute the
   median of ``local_recv_ts_ns/1e6 − exchange_ts_ms`` over all observed
   trades. Subtracting per-exchange ``offset − consensus_median`` from each
   trade's ``timestamp_ms`` removes systematic differences between exchange
   matching-engine clocks (which can disagree by 100s of ms, sometimes
   seconds) so that a phantom lag does not appear simply because exchange A
   stamps trades 200 ms earlier than exchange B.

2. **Resample last-trade prices on a uniform grid** (default 1 second).
   Bars containing no trade are forward-filled *for the price* but tracked
   as *inactive* in a parallel boolean mask so the correlation can avoid
   them.

3. **Compute log-returns** on each side, restricted to bars where *both*
   sides actually saw a trade in the bar (the active mask). This is the
   key fix for low-tick-frequency exchanges: a venue that prints once a
   minute contributes ~1 active bar per minute and a thin slice of stale
   forward-fill, instead of 60 ffill bars that look correlated to whoever
   they happen to align with.

4. **Drop low-tick (exchange, symbol)** combinations where the active rate
   (active bars / total bars) falls below ``min_active_rate``. Such
   combinations otherwise produce spurious "lag" estimates from staircase
   aliasing of forward-fill.

5. **Score correlation across the lag grid** and locate the lag whose
   absolute correlation is largest. The leader is the side whose returns
   *lead* the other.

6. **Estimate a confidence interval** on the optimal lag via block-bootstrap
   that re-applies the active mask on each draw.

The analyser is deterministic given a fixed ``rng``; callers wanting
reproducibility should pass a ``numpy.random.Generator`` into
:func:`analyze_all` or :func:`bootstrap_lag_ci`.

Backwards compatibility: callers passing an :class:`AnalyzerConfig` with
``active_mask=False`` and ``clock_skew_calibration=False`` get the legacy
forward-fill / un-calibrated behaviour, which is useful for sanity-checking
the impact of each fix independently.
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
    active_rate_a: float = 1.0  # fraction of overlap bars where exchange_a saw a trade
    active_rate_b: float = 1.0
    n_co_active: int = 0  # bars where both exchanges had real trades, post-mask


@dataclass(frozen=True, slots=True)
class ExchangeDiagnostics:
    """Per-exchange data-quality statistics surfaced in the report.

    ``transport_latency_p50_ms`` is the median of
    ``local_recv_ts_ns/1e6 − exchange_ts_ms`` and includes both network RTT
    and any clock skew between the exchange's matching engine and our local
    clock. The analyzer subtracts the cross-exchange median of these values
    from each timestamp, so only the *relative* differences end up shifting
    the data.
    """

    exchange: str
    n_trades: int
    transport_latency_p50_ms: float  # median local_recv - exchange_ts
    clock_offset_ms: float  # offset applied during calibration


def compute_clock_offsets(
    trades: pd.DataFrame,
    *,
    max_drift_seconds: float = 3600.0,
) -> tuple[dict[str, float], list[ExchangeDiagnostics]]:
    """Return per-exchange clock offsets (in ms) and diagnostics.

    The offset for exchange ``E`` is the median of
    ``local_recv_ts_ns / 1e6 − exchange_ts_ms`` across all of ``E``'s
    trades that have a non-zero ``local_recv_ts_ns``. The *consensus
    median* across exchanges is subtracted before applying the offset, so
    only the *relative* skew between exchanges shifts the data.

    Trades with ``local_recv_ts_ns == 0`` (legacy shards) or whose drift
    exceeds ``max_drift_seconds`` are excluded from the median.

    The returned ``offsets`` map values that are ready to be *added* to
    ``timestamp_ms`` (so a positive offset means "this exchange's stamps
    are early relative to consensus and should be pushed forward").
    """

    if trades.empty or "local_recv_ts_ns" not in trades.columns:
        return {}, []

    exchange_arr = np.asarray(trades["exchange"].to_numpy())
    ts_ms_arr = np.asarray(trades["timestamp_ms"].to_numpy(), dtype="int64")
    local_ns_arr = np.asarray(trades["local_recv_ts_ns"].to_numpy(), dtype="int64")
    has_local = local_ns_arr > 0
    if not bool(has_local.any()):
        return {}, []
    exchange_arr = exchange_arr[has_local]
    ts_ms_arr = ts_ms_arr[has_local]
    local_ns_arr = local_ns_arr[has_local]

    delta_ms = (local_ns_arr // 1_000_000 - ts_ms_arr).astype("float64")
    drift_cap_ms = max_drift_seconds * 1000.0
    in_window = np.abs(delta_ms) <= drift_cap_ms
    if not bool(in_window.any()):
        return {}, []
    exchange_arr = exchange_arr[in_window]
    delta_ms = delta_ms[in_window]

    base = pd.DataFrame({"exchange": exchange_arr, "delta_ms": delta_ms})
    grp = base.groupby("exchange")["delta_ms"]
    medians = grp.median()
    sizes = grp.size()
    if medians.empty:
        return {}, []

    medians_np = np.asarray(medians.to_numpy(), dtype="float64")
    sizes_np = np.asarray(sizes.to_numpy(), dtype="int64")
    consensus = float(np.median(medians_np))
    diags: list[ExchangeDiagnostics] = []
    offsets: dict[str, float] = {}
    for i, ex_idx in enumerate(medians.index):
        ex = str(ex_idx)
        median_ms = float(medians_np[i])
        n = int(sizes_np[i])
        # Adding ``consensus - median_ms`` to ``timestamp_ms`` re-centers
        # this exchange to the consensus timeline. Exchanges whose stamps
        # are *late* relative to the median (large positive median) get a
        # negative offset (pulled earlier).
        applied = consensus - median_ms
        offsets[ex] = applied
        diags.append(
            ExchangeDiagnostics(
                exchange=ex,
                n_trades=n,
                transport_latency_p50_ms=median_ms,
                clock_offset_ms=applied,
            )
        )
    diags.sort(key=lambda d: d.exchange)
    return offsets, diags


def _resample_with_mask(
    timestamp_ms: np.ndarray,
    price: np.ndarray,
    resample_seconds: float,
) -> tuple[pd.Series, pd.Series]:
    """Resample to a uniform grid and return ``(prices, active_mask)``.

    ``prices`` is forward-filled so it can be diffed; ``active_mask`` is
    ``True`` only on bars whose ``last()`` aggregation saw a real trade
    (i.e. not a forward-filled bar).
    """

    if timestamp_ms.size == 0:
        idx = pd.DatetimeIndex([], tz="UTC")
        return pd.Series(dtype="float64", index=idx), pd.Series(dtype="bool", index=idx)
    series = pd.Series(
        price.astype("float64"),
        index=pd.to_datetime(timestamp_ms, unit="ms", utc=True),
        name="price",
    ).sort_index()
    rule = f"{int(resample_seconds * 1000)}ms"
    last = series.resample(rule).last()
    active_raw = last.notna()
    prices = last.ffill().dropna()
    if prices.empty:
        return prices, pd.Series(dtype="bool", index=prices.index)
    active = active_raw.reindex(prices.index, fill_value=False).astype(bool)
    return prices, active


def _aligned_returns(
    pa_: pd.Series,
    pb_: pd.Series,
    ma_: pd.Series | None = None,
    mb_: pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute log-returns on the intersection of two price series.

    Returns ``(ra, rb, a_active, b_active)``. The active arrays are aligned
    to the *return* indices: a return at index ``i`` is "active" on side
    ``A`` if ``A`` had a real trade in the bar that produced the return
    (i.e. it's backed by a fresh trade rather than forward-fill).

    When the two mask Series are ``None`` the returned masks are all-True,
    preserving the legacy un-masked behaviour.
    """

    if pa_.empty or pb_.empty:
        empty = np.array([])
        empty_b = np.array([], dtype=bool)
        return empty, empty, empty_b, empty_b
    common_index = pa_.index.intersection(pb_.index)
    if len(common_index) < 3:
        empty = np.array([])
        empty_b = np.array([], dtype=bool)
        return empty, empty, empty_b, empty_b
    a_aligned = pa_.reindex(common_index)
    b_aligned = pb_.reindex(common_index)
    ra = np.diff(np.log(a_aligned.to_numpy(dtype="float64")))
    rb = np.diff(np.log(b_aligned.to_numpy(dtype="float64")))
    if ma_ is None or mb_ is None:
        ones = np.ones(ra.size, dtype=bool)
        return ra, rb, ones, ones
    a_active_full = ma_.reindex(common_index, fill_value=False).to_numpy(dtype=bool)
    b_active_full = mb_.reindex(common_index, fill_value=False).to_numpy(dtype=bool)
    a_active = a_active_full[1:]
    b_active = b_active_full[1:]
    return ra, rb, a_active, b_active


def _lag_grid_array(grid: LagGrid, resample_seconds: float) -> np.ndarray:
    """Return the lag grid in *integer steps of resample_seconds*.

    A lag of 0.5s with resample_seconds=0.1s ⇒ +5 steps.
    Non-multiples of resample_seconds are rounded to the nearest integer step.
    """

    raw = np.arange(grid.start, grid.stop + grid.step / 2, grid.step)
    steps = np.round(raw / resample_seconds).astype(int)
    return np.unique(steps)


def _slice_for_lag(
    arr: np.ndarray,
    side: str,
    lag: int,
    n: int,
) -> np.ndarray:
    """Return the slice of ``arr`` used for the given lag.

    ``side`` is ``"a"`` for the leader array and ``"b"`` for the follower
    (which is shifted forward by ``lag``).
    """

    if lag >= 0:
        return arr[: n - lag] if side == "a" else arr[lag : lag + (n - lag)]
    k = -lag
    return arr[k : k + (n - k)] if side == "a" else arr[: n - k]


def _pair_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    sx = float(x.std(ddof=1))
    sy = float(y.std(ddof=1))
    if sx == 0 or sy == 0 or not np.isfinite(sx) or not np.isfinite(sy):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def cross_correlation(
    ra: np.ndarray,
    rb: np.ndarray,
    lag_steps: np.ndarray,
    a_active: np.ndarray | None = None,
    b_active: np.ndarray | None = None,
) -> np.ndarray:
    """Compute Pearson correlation between ``ra`` and ``rb`` shifted by ``lag``.

    Positive lag ⇒ ``rb`` is shifted forward in time relative to ``ra``,
    i.e. ``corr[i] = corr(ra[t], rb[t + lag_steps[i]])``.

    When ``a_active`` and ``b_active`` are provided, only sample positions
    where *both* sides had a real trade in their respective bar are used.
    This is the active-mask path that protects against forward-fill bias on
    low-tick exchanges.

    Returns an array of correlations of the same length as ``lag_steps``.
    NaN is returned when the overlap shrinks below 3 points or variance
    collapses to zero.
    """

    if ra.size == 0 or rb.size == 0:
        return np.full(lag_steps.shape, np.nan)
    out = np.empty(lag_steps.shape, dtype="float64")
    n = min(ra.size, rb.size)
    if a_active is not None and b_active is not None:
        ma = a_active
        mb = b_active
        for i, lag in enumerate(lag_steps):
            x = _slice_for_lag(ra, "a", int(lag), n)
            y = _slice_for_lag(rb, "b", int(lag), n)
            xa = _slice_for_lag(ma, "a", int(lag), n)
            ya = _slice_for_lag(mb, "b", int(lag), n)
            both = xa & ya
            out[i] = _pair_corr(x[both], y[both])
    else:
        for i, lag in enumerate(lag_steps):
            x = _slice_for_lag(ra, "a", int(lag), n)
            y = _slice_for_lag(rb, "b", int(lag), n)
            out[i] = _pair_corr(x, y)
    return out


def bootstrap_lag_ci(
    ra: np.ndarray,
    rb: np.ndarray,
    lag_steps: np.ndarray,
    config: BootstrapConfig,
    *,
    a_active: np.ndarray | None = None,
    b_active: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Block-bootstrap a 95% CI on the argmax-|corr| lag (in *steps*).

    Returns ``(low_steps, high_steps)``. When active masks are given they
    are resampled together with ``ra`` / ``rb`` so each bootstrap draw
    re-applies the same masking discipline as the point estimate.
    """

    if rng is None:
        rng = np.random.default_rng(0)
    n = min(ra.size, rb.size)
    if n < config.block_size * 3:
        return float("nan"), float("nan")
    block_starts = np.arange(0, n - config.block_size + 1)
    n_blocks = max(1, n // config.block_size)
    best_lags: list[int] = []
    ma = a_active if a_active is not None and b_active is not None else None
    mb = b_active if a_active is not None and b_active is not None else None
    for _ in range(config.n_iter):
        chosen = rng.choice(block_starts, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, s + config.block_size) for s in chosen])
        idx = idx[idx < n]
        if ma is not None and mb is not None:
            sample_corrs = cross_correlation(ra[idx], rb[idx], lag_steps, ma[idx], mb[idx])
        else:
            sample_corrs = cross_correlation(ra[idx], rb[idx], lag_steps)
        if np.all(np.isnan(sample_corrs)):
            continue
        best_lags.append(int(lag_steps[np.nanargmax(np.abs(sample_corrs))]))
    if not best_lags:
        return float("nan"), float("nan")
    arr = np.array(best_lags, dtype="float64")
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def _adjusted_timestamp_ms(
    df: pd.DataFrame,
    exchange: str,
    offsets: dict[str, float] | None,
) -> np.ndarray:
    """Return ``timestamp_ms`` with the per-exchange clock-skew offset added."""

    raw = df["timestamp_ms"].to_numpy(dtype="int64")
    if not offsets or exchange not in offsets:
        return raw
    offset = round(offsets[exchange])
    if offset == 0:
        return raw
    return raw + offset


@dataclass(frozen=True, slots=True)
class _PreparedSeries:
    """Resampled prices, active masks, and active-rate stats for a pair."""

    pa_: pd.Series
    pb_: pd.Series
    ma_: pd.Series
    mb_: pd.Series
    active_rate_a: float
    active_rate_b: float


def _prepare_series(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    exchange_a: str,
    exchange_b: str,
    config: AnalyzerConfig,
    offsets: dict[str, float] | None,
) -> _PreparedSeries | None:
    use_offsets = config.clock_skew_calibration and offsets is not None
    if use_offsets:
        ts_a = _adjusted_timestamp_ms(df_a, exchange_a, offsets)
        ts_b = _adjusted_timestamp_ms(df_b, exchange_b, offsets)
    else:
        ts_a = df_a["timestamp_ms"].to_numpy(dtype="int64")
        ts_b = df_b["timestamp_ms"].to_numpy(dtype="int64")
    pa_, ma_ = _resample_with_mask(ts_a, df_a["price"].to_numpy(), config.resample_seconds)
    pb_, mb_ = _resample_with_mask(ts_b, df_b["price"].to_numpy(), config.resample_seconds)
    if pa_.empty or pb_.empty:
        return None
    common_index = pa_.index.intersection(pb_.index)
    if len(common_index) == 0:
        return None
    n_common = len(common_index)
    a_active_count = int(ma_.reindex(common_index, fill_value=False).sum())
    b_active_count = int(mb_.reindex(common_index, fill_value=False).sum())
    active_rate_a = a_active_count / n_common
    active_rate_b = b_active_count / n_common
    return _PreparedSeries(
        pa_=pa_,
        pb_=pb_,
        ma_=ma_,
        mb_=mb_,
        active_rate_a=active_rate_a,
        active_rate_b=active_rate_b,
    )


def _follower_std(
    follower: str,
    exchange_b: str,
    ra: np.ndarray,
    rb: np.ndarray,
    a_active: np.ndarray,
    b_active: np.ndarray,
    use_mask: bool,
) -> float:
    follower_returns = rb if follower == exchange_b else ra
    if follower_returns.size <= 1:
        return 0.0
    if use_mask:
        follower_active = b_active if follower == exchange_b else a_active
        follower_masked = follower_returns[follower_active]
        if follower_masked.size <= 1:
            return 0.0
        return float(follower_masked.std(ddof=1))
    return float(follower_returns.std(ddof=1))


def analyze_pair(
    symbol: str,
    exchange_a: str,
    exchange_b: str,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    config: AnalyzerConfig,
    *,
    offsets: dict[str, float] | None = None,
    rng: np.random.Generator | None = None,
) -> LeadLagResult | None:
    """Analyse a single ``(symbol, exchange_a, exchange_b)`` triple.

    Returns ``None`` if the pair has fewer than ``config.min_obs`` overlapping
    co-active observations, returns are degenerate, or either side falls
    below ``config.min_active_rate`` (when ``active_mask`` is on).
    """

    prep = _prepare_series(df_a, df_b, exchange_a, exchange_b, config, offsets)
    if prep is None:
        return None
    if config.active_mask and (
        prep.active_rate_a < config.min_active_rate or prep.active_rate_b < config.min_active_rate
    ):
        return None

    use_mask = config.active_mask
    if use_mask:
        ra, rb, a_active, b_active = _aligned_returns(prep.pa_, prep.pb_, prep.ma_, prep.mb_)
    else:
        ra, rb, a_active, b_active = _aligned_returns(prep.pa_, prep.pb_)
    n_returns = min(ra.size, rb.size)
    if n_returns < 3:
        return None

    lag_steps = _lag_grid_array(config.lag_grid, config.resample_seconds)
    corrs = (
        cross_correlation(ra, rb, lag_steps, a_active, b_active)
        if use_mask
        else cross_correlation(ra, rb, lag_steps)
    )
    if np.all(np.isnan(corrs)):
        return None

    co_active_count = int((a_active & b_active).sum()) if use_mask else n_returns
    effective_obs = co_active_count if use_mask else n_returns
    if effective_obs < config.min_obs:
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

    ci_low_steps, ci_high_steps = bootstrap_lag_ci(
        ra,
        rb,
        lag_steps,
        config.bootstrap,
        a_active=a_active if use_mask else None,
        b_active=b_active if use_mask else None,
        rng=rng,
    )

    return LeadLagResult(
        symbol=symbol,
        exchange_a=exchange_a,
        exchange_b=exchange_b,
        n_obs=effective_obs,
        best_lag_seconds=best_lag_seconds,
        best_correlation=best_corr,
        correlation_at_zero=corr_at_zero,
        leader=leader,
        follower=follower,
        lag_ci_low_seconds=ci_low_steps * config.resample_seconds,
        lag_ci_high_seconds=ci_high_steps * config.resample_seconds,
        follower_return_std=_follower_std(
            follower, exchange_b, ra, rb, a_active, b_active, use_mask
        ),
        active_rate_a=prep.active_rate_a,
        active_rate_b=prep.active_rate_b,
        n_co_active=co_active_count,
    )


@dataclass(frozen=True, slots=True)
class AnalysisOutput:
    """Bundle returned by :func:`analyze_all_with_diagnostics`.

    The plain :func:`analyze_all` returns just the list of results for
    backwards compatibility.
    """

    results: list[LeadLagResult]
    diagnostics: list[ExchangeDiagnostics]


def analyze_all_with_diagnostics(
    trades: pd.DataFrame,
    config: AnalyzerConfig,
    *,
    rng: np.random.Generator | None = None,
) -> AnalysisOutput:
    """Run :func:`analyze_pair` over every (symbol, exchange-pair) combination
    present in ``trades``, also returning per-exchange diagnostics.
    """

    if trades.empty:
        return AnalysisOutput(results=[], diagnostics=[])

    if config.clock_skew_calibration:
        offsets, diagnostics = compute_clock_offsets(
            trades, max_drift_seconds=config.max_clock_drift_seconds
        )
    else:
        offsets, diagnostics = {}, []

    results: list[LeadLagResult] = []
    by_symbol: dict[str, dict[str, pd.DataFrame]] = {}
    for symbol_val, sym_group in trades.groupby("symbol", sort=True):
        symbol = str(symbol_val)
        by_symbol[symbol] = {}
        for exchange_val, ex_group in sym_group.groupby("exchange", sort=True):
            by_symbol[symbol][str(exchange_val)] = ex_group

    for _symbol, by_exchange in by_symbol.items():
        exchanges = sorted(by_exchange.keys())
        for ex_a, ex_b in combinations(exchanges, 2):
            result = analyze_pair(
                _symbol,
                ex_a,
                ex_b,
                by_exchange[ex_a],
                by_exchange[ex_b],
                config,
                offsets=offsets,
                rng=rng,
            )
            if result is not None:
                results.append(result)
    return AnalysisOutput(results=results, diagnostics=diagnostics)


def analyze_all(
    trades: pd.DataFrame,
    config: AnalyzerConfig,
    *,
    rng: np.random.Generator | None = None,
) -> list[LeadLagResult]:
    """Run :func:`analyze_pair` over every (symbol, exchange-pair) combination
    present in ``trades``.

    Thin wrapper over :func:`analyze_all_with_diagnostics` that discards the
    diagnostics. Prefer the latter when you want to surface per-exchange
    clock-skew information in a report.
    """

    return analyze_all_with_diagnostics(trades, config, rng=rng).results
