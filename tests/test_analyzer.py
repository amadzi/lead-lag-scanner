"""Property tests for the lead-lag analyser using synthetic data with known lag.

We construct a leader price series via a random walk, then build a follower
series that copies the leader's last log-return after a known delay. The
analyser must (a) recover that delay and (b) assign leader/follower correctly.

This module also exercises the four data-quality fixes:
  * clock-skew calibration via :func:`compute_clock_offsets`
  * sanity filter (drift cap) inside :func:`compute_clock_offsets`
  * active-bar masking via :func:`_resample_with_mask` and the masked
    :func:`cross_correlation` path
  * low-tick filtering inside :func:`analyze_pair`
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from lead_lag_scanner.analyzer import (
    LeadLagResult,
    _aligned_returns,
    _resample_with_mask,
    analyze_all,
    analyze_all_with_diagnostics,
    analyze_pair,
    compute_clock_offsets,
    cross_correlation,
)
from lead_lag_scanner.config import AnalyzerConfig, BootstrapConfig, LagGrid


def _synthetic_trades(
    *,
    n_seconds: int,
    leader_exchange: str,
    follower_exchange: str,
    symbol: str,
    lag_seconds: int,
    seed: int = 42,
    follower_noise: float = 1e-5,
    include_local_recv: bool = False,
    leader_clock_offset_ms: int = 0,
    follower_clock_offset_ms: int = 0,
) -> pd.DataFrame:
    """Create synthetic trades on a 1-second grid with a known lag.

    Leader is a Geometric Brownian motion. Follower copies leader at t-lag
    with a small additive noise. When ``include_local_recv`` is set, every
    trade is given a ``local_recv_ts_ns`` field equal to ``timestamp_ms``
    in nanoseconds (no transport latency); the per-side
    ``*_clock_offset_ms`` arguments shift just that exchange's reported
    ``timestamp_ms`` so we can simulate clock drift.
    """

    rng = np.random.default_rng(seed)
    base_ts_ms = 1_700_000_000_000
    log_returns = rng.normal(0.0, 1e-4, size=n_seconds)
    leader_prices = 50_000.0 * np.exp(np.cumsum(log_returns))

    follower_prices = np.empty_like(leader_prices)
    for t in range(n_seconds):
        src = max(0, t - lag_seconds)
        follower_prices[t] = leader_prices[src] * (1.0 + rng.normal(0.0, follower_noise))

    rows: list[dict[str, object]] = []
    for t in range(n_seconds):
        ts_ms = base_ts_ms + t * 1000
        leader_row: dict[str, object] = {
            "timestamp_ms": ts_ms + leader_clock_offset_ms,
            "exchange": leader_exchange,
            "symbol": symbol,
            "price": float(leader_prices[t]),
            "amount": 0.01,
            "side": "buy",
            "trade_id": f"L-{t}",
        }
        follower_row: dict[str, object] = {
            "timestamp_ms": ts_ms + follower_clock_offset_ms,
            "exchange": follower_exchange,
            "symbol": symbol,
            "price": float(follower_prices[t]),
            "amount": 0.01,
            "side": "sell",
            "trade_id": f"F-{t}",
        }
        if include_local_recv:
            leader_row["local_recv_ts_ns"] = ts_ms * 1_000_000
            follower_row["local_recv_ts_ns"] = ts_ms * 1_000_000
        rows.append(leader_row)
        rows.append(follower_row)
    return pd.DataFrame(rows)


def _config_for_test(
    *,
    lag_seconds_max: float = 5.0,
    min_obs: int = 50,
    n_iter: int = 20,
    active_mask: bool = False,
    clock_skew_calibration: bool = False,
    min_active_rate: float = 0.0,
) -> AnalyzerConfig:
    """A test config with the four fixes off by default for legacy tests."""

    return AnalyzerConfig(
        resample_seconds=1.0,
        lag_grid=LagGrid(start=-lag_seconds_max, stop=lag_seconds_max, step=1.0),
        min_obs=min_obs,
        bootstrap=BootstrapConfig(block_size=10, n_iter=n_iter),
        active_mask=active_mask,
        clock_skew_calibration=clock_skew_calibration,
        min_active_rate=min_active_rate,
    )


# _resample_with_mask ------------------------------------------------------


def test_resample_with_mask_handles_empty() -> None:
    prices, active = _resample_with_mask(
        np.array([], dtype="int64"), np.array([], dtype="float64"), 1.0
    )
    assert prices.empty
    assert active.empty


def test_resample_with_mask_marks_forward_filled_bars_inactive() -> None:
    ts = np.array([0, 3000, 5000], dtype="int64")
    px = np.array([100.0, 110.0, 105.0], dtype="float64")
    prices, active = _resample_with_mask(ts, px, 1.0)
    # 6 bars (0..5s), forward-filled.
    assert len(prices) == 6
    assert prices.iloc[0] == 100.0
    assert prices.iloc[2] == 100.0
    assert prices.iloc[3] == 110.0
    assert prices.iloc[5] == 105.0
    # Active mask is True only at the bars with a real trade.
    assert bool(active.iloc[0]) is True
    assert bool(active.iloc[1]) is False
    assert bool(active.iloc[2]) is False
    assert bool(active.iloc[3]) is True
    assert bool(active.iloc[4]) is False
    assert bool(active.iloc[5]) is True


# cross_correlation -------------------------------------------------------


def test_cross_correlation_recovers_known_lag() -> None:
    rng = np.random.default_rng(0)
    n = 1000
    ra = rng.normal(0.0, 1.0, size=n)
    # rb at t == ra at (t - 3); positive lag => ra leads rb.
    rb = np.concatenate([np.zeros(3), ra[:-3]])

    lag_steps = np.arange(-5, 6)
    corrs = cross_correlation(ra, rb, lag_steps)
    best_lag = int(lag_steps[np.nanargmax(np.abs(corrs))])
    assert best_lag == 3


def test_cross_correlation_with_active_mask_changes_result() -> None:
    """Active mask should change which bars contribute to the correlation."""

    rng = np.random.default_rng(1)
    n = 500
    ra = rng.normal(0.0, 1.0, size=n)
    rb = rng.normal(0.0, 1.0, size=n)
    # Force 90% of side A's bars to be inactive (forward-filled).
    a_active = np.zeros(n, dtype=bool)
    a_active[::10] = True
    b_active = np.ones(n, dtype=bool)

    lag_steps = np.arange(-3, 4)
    masked = cross_correlation(ra, rb, lag_steps, a_active, b_active)
    unmasked = cross_correlation(ra, rb, lag_steps)
    assert np.isfinite(masked).any()
    # Masking actually changed the correlation (didn't silently fall through).
    assert not np.allclose(masked, unmasked, equal_nan=True)


def test_aligned_returns_handles_mismatched_indices() -> None:
    a = pd.Series(
        [100, 101, 102, 103],
        index=pd.to_datetime([1_000, 2_000, 3_000, 4_000], unit="ms", utc=True),
    )
    b = pd.Series(
        [200, 201, 202],
        index=pd.to_datetime([2_000, 3_000, 4_000], unit="ms", utc=True),
    )
    ra, rb, a_act, b_act = _aligned_returns(a, b)
    assert ra.shape == rb.shape
    assert ra.shape[0] == 2  # 3 aligned points -> 2 returns
    assert a_act.shape == ra.shape
    assert bool(a_act.all())
    assert bool(b_act.all())


# compute_clock_offsets ---------------------------------------------------


def test_compute_clock_offsets_returns_empty_when_no_local_recv() -> None:
    df = _synthetic_trades(
        n_seconds=60,
        leader_exchange="binance",
        follower_exchange="kucoin",
        symbol="BTC/USDT",
        lag_seconds=1,
        include_local_recv=False,
    )
    df["local_recv_ts_ns"] = 0
    offsets, diags = compute_clock_offsets(df)
    assert offsets == {}
    assert diags == []


def test_compute_clock_offsets_recovers_relative_skew() -> None:
    """A 500 ms early stamp on follower yields a non-zero relative offset."""

    df = _synthetic_trades(
        n_seconds=120,
        leader_exchange="binance",
        follower_exchange="kucoin",
        symbol="BTC/USDT",
        lag_seconds=0,
        include_local_recv=True,
        leader_clock_offset_ms=0,
        follower_clock_offset_ms=-500,
    )
    offsets, diags = compute_clock_offsets(df)
    assert set(offsets.keys()) == {"binance", "kucoin"}
    spread = max(offsets.values()) - min(offsets.values())
    assert spread == pytest.approx(500.0, abs=5.0)
    assert {d.exchange for d in diags} == {"binance", "kucoin"}


def test_compute_clock_offsets_drops_drift_outliers() -> None:
    df = _synthetic_trades(
        n_seconds=120,
        leader_exchange="binance",
        follower_exchange="kucoin",
        symbol="BTC/USDT",
        lag_seconds=0,
        include_local_recv=True,
    )
    # Inject one absurdly off timestamp on each side (1 day off).
    df.loc[0, "timestamp_ms"] = int(df.loc[0, "timestamp_ms"]) - 24 * 3600 * 1000
    df.loc[1, "timestamp_ms"] = int(df.loc[1, "timestamp_ms"]) + 24 * 3600 * 1000
    offsets, _ = compute_clock_offsets(df, max_drift_seconds=3600.0)
    # Without the cap the median would be dragged off zero.
    assert all(abs(v) < 50.0 for v in offsets.values())


# analyze_pair ------------------------------------------------------------


def test_analyze_pair_recovers_leader_follower() -> None:
    df = _synthetic_trades(
        n_seconds=600,
        leader_exchange="binance",
        follower_exchange="kucoin",
        symbol="BTC/USDT",
        lag_seconds=2,
    )
    cfg = _config_for_test()
    df_a = cast(pd.DataFrame, df[df["exchange"] == "binance"])
    df_b = cast(pd.DataFrame, df[df["exchange"] == "kucoin"])
    result = analyze_pair("BTC/USDT", "binance", "kucoin", df_a, df_b, cfg)
    assert result is not None
    assert isinstance(result, LeadLagResult)
    assert result.leader == "binance"
    assert result.follower == "kucoin"
    assert result.best_lag_seconds == pytest.approx(2.0, abs=1.0)
    assert abs(result.best_correlation) > 0.5


def test_analyze_pair_swapped_order_still_recovers_leader() -> None:
    df = _synthetic_trades(
        n_seconds=600,
        leader_exchange="binance",
        follower_exchange="kucoin",
        symbol="BTC/USDT",
        lag_seconds=2,
    )
    cfg = _config_for_test()
    df_a = cast(pd.DataFrame, df[df["exchange"] == "kucoin"])
    df_b = cast(pd.DataFrame, df[df["exchange"] == "binance"])
    result = analyze_pair("BTC/USDT", "kucoin", "binance", df_a, df_b, cfg)
    assert result is not None
    assert result.leader == "binance"
    assert result.follower == "kucoin"
    assert result.best_lag_seconds < 0


def test_analyze_pair_returns_none_below_min_obs() -> None:
    df = _synthetic_trades(
        n_seconds=20,
        leader_exchange="binance",
        follower_exchange="kucoin",
        symbol="BTC/USDT",
        lag_seconds=1,
    )
    cfg = _config_for_test(min_obs=1000)
    df_a = cast(pd.DataFrame, df[df["exchange"] == "binance"])
    df_b = cast(pd.DataFrame, df[df["exchange"] == "kucoin"])
    assert analyze_pair("BTC/USDT", "binance", "kucoin", df_a, df_b, cfg) is None


def test_analyze_pair_low_tick_drop_with_active_mask() -> None:
    """An exchange with very few trades must be dropped under the active mask."""

    df = _synthetic_trades(
        n_seconds=600,
        leader_exchange="binance",
        follower_exchange="kucoin",
        symbol="BTC/USDT",
        lag_seconds=1,
    )
    df_a = cast(pd.DataFrame, df[df["exchange"] == "binance"])
    df_b = cast(pd.DataFrame, df[df["exchange"] == "kucoin"])
    df_b_sparse = df_b.iloc[::50].reset_index(drop=True)
    cfg = _config_for_test(active_mask=True, min_active_rate=0.05, min_obs=50)
    result = analyze_pair("BTC/USDT", "binance", "kucoin", df_a, df_b_sparse, cfg)
    assert result is None


def test_analyze_pair_active_mask_path_recovers_lag() -> None:
    df = _synthetic_trades(
        n_seconds=600,
        leader_exchange="binance",
        follower_exchange="kucoin",
        symbol="BTC/USDT",
        lag_seconds=2,
    )
    df_a = cast(pd.DataFrame, df[df["exchange"] == "binance"])
    df_b = cast(pd.DataFrame, df[df["exchange"] == "kucoin"])
    cfg = _config_for_test(active_mask=True, min_active_rate=0.05)
    result = analyze_pair("BTC/USDT", "binance", "kucoin", df_a, df_b, cfg)
    assert result is not None
    assert result.leader == "binance"
    assert result.follower == "kucoin"
    assert result.active_rate_a > 0.9
    assert result.active_rate_b > 0.9
    assert result.n_co_active > 0


# analyze_all / analyze_all_with_diagnostics ------------------------------


def test_analyze_all_runs_over_all_pairs() -> None:
    parts = [
        _synthetic_trades(
            n_seconds=400,
            leader_exchange="binance",
            follower_exchange="kucoin",
            symbol="BTC/USDT",
            lag_seconds=1,
            seed=1,
        ),
        _synthetic_trades(
            n_seconds=400,
            leader_exchange="binance",
            follower_exchange="okx",
            symbol="ETH/USDT",
            lag_seconds=2,
            seed=2,
        ),
    ]
    df = pd.concat(parts, ignore_index=True)
    cfg = _config_for_test()
    results = analyze_all(df, cfg)
    assert len(results) >= 2

    btc_results = [r for r in results if r.symbol == "BTC/USDT"]
    eth_results = [r for r in results if r.symbol == "ETH/USDT"]
    assert btc_results
    assert eth_results
    assert btc_results[0].leader == "binance"
    assert eth_results[0].leader == "binance"


def test_analyze_all_with_diagnostics_returns_per_exchange_diags() -> None:
    df = _synthetic_trades(
        n_seconds=400,
        leader_exchange="binance",
        follower_exchange="kucoin",
        symbol="BTC/USDT",
        lag_seconds=1,
        include_local_recv=True,
        follower_clock_offset_ms=-200,
    )
    cfg = _config_for_test(clock_skew_calibration=True)
    out = analyze_all_with_diagnostics(df, cfg)
    assert {d.exchange for d in out.diagnostics} == {"binance", "kucoin"}
    by_ex = {d.exchange: d for d in out.diagnostics}
    # The follower stamps 200 ms early -> clock_offset_ms negative there.
    assert by_ex["kucoin"].clock_offset_ms < 0
    assert by_ex["binance"].clock_offset_ms > 0


def test_analyze_all_handles_empty() -> None:
    cfg = _config_for_test()
    results = analyze_all(pd.DataFrame(), cfg)
    assert results == []


def test_analyze_all_skips_pair_that_raises_unexpected_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single pathological (symbol, exchange-pair) must not kill the whole run.

    The user reported a ``analyze: operands could not be broadcast together
    with shapes (0,) (3,)`` error in the dashboard. The original
    ``analyze_all_with_diagnostics`` did not have a per-pair try/except so
    one such failure took down every other pair too. We now skip the bad
    pair and continue. This test forces ``analyze_pair`` to raise on the
    first invocation (mimicking the actual error wording the user saw)
    and asserts that the second pair still produces a result.
    """

    df_btc = _synthetic_trades(
        n_seconds=600,
        leader_exchange="binance",
        follower_exchange="okx",
        symbol="BTC/USDT",
        lag_seconds=2,
    )
    df_eth = _synthetic_trades(
        n_seconds=600,
        leader_exchange="binance",
        follower_exchange="okx",
        symbol="ETH/USDT",
        lag_seconds=2,
        seed=99,
    )
    trades = pd.concat([df_btc, df_eth], ignore_index=True)
    config = _config_for_test()

    real_pair = analyze_pair
    call_count = {"n": 0}

    def flaky_pair(*args: object, **kwargs: object) -> LeadLagResult | None:
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Mimic numpy's actual error wording so future searches still
            # match this test if the dashboard surfaces a similar one.
            raise ValueError("operands could not be broadcast together with shapes (0,) (3,) ")
        return cast(LeadLagResult | None, real_pair(*args, **kwargs))  # type: ignore[arg-type]

    monkeypatch.setattr("lead_lag_scanner.analyzer.analyze_pair", flaky_pair)

    out = analyze_all_with_diagnostics(trades, config)
    assert call_count["n"] == 2
    assert len(out.results) == 1
    assert out.results[0].symbol in {"BTC/USDT", "ETH/USDT"}
