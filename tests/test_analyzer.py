"""Property tests for the lead-lag analyser using synthetic data with known lag.

We construct a leader price series via a random walk, then build a follower
series that copies the leader's last log-return after a known delay. The
analyser must (a) recover that delay and (b) assign leader/follower correctly.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from lead_lag_scanner.analyzer import (
    LeadLagResult,
    _aligned_returns,
    _resample_last_price,
    analyze_all,
    analyze_pair,
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
) -> pd.DataFrame:
    """Create synthetic trades on a 1-second grid with a known lag.

    Leader is a Geometric Brownian motion. Follower copies leader at t-lag with
    a small additive noise.
    """

    rng = np.random.default_rng(seed)
    base_ts_ms = 1_700_000_000_000
    log_returns = rng.normal(0.0, 1e-4, size=n_seconds)
    leader_prices = 50_000.0 * np.exp(np.cumsum(log_returns))

    # Follower at time t reflects leader at time (t - lag), plus tiny noise.
    follower_prices = np.empty_like(leader_prices)
    for t in range(n_seconds):
        src = max(0, t - lag_seconds)
        follower_prices[t] = leader_prices[src] * (1.0 + rng.normal(0.0, follower_noise))

    rows: list[dict[str, object]] = []
    for t in range(n_seconds):
        ts_ms = base_ts_ms + t * 1000
        rows.append(
            {
                "timestamp_ms": ts_ms,
                "exchange": leader_exchange,
                "symbol": symbol,
                "price": float(leader_prices[t]),
                "amount": 0.01,
                "side": "buy",
                "trade_id": f"L-{t}",
            }
        )
        rows.append(
            {
                "timestamp_ms": ts_ms,
                "exchange": follower_exchange,
                "symbol": symbol,
                "price": float(follower_prices[t]),
                "amount": 0.01,
                "side": "sell",
                "trade_id": f"F-{t}",
            }
        )
    return pd.DataFrame(rows)


def _config_for_test(
    *, lag_seconds_max: float = 5.0, min_obs: int = 50, n_iter: int = 20
) -> AnalyzerConfig:
    return AnalyzerConfig(
        resample_seconds=1.0,
        lag_grid=LagGrid(start=-lag_seconds_max, stop=lag_seconds_max, step=1.0),
        min_obs=min_obs,
        bootstrap=BootstrapConfig(block_size=10, n_iter=n_iter),
    )


def test_resample_last_price_handles_empty() -> None:
    s = _resample_last_price(pd.DataFrame(columns=["timestamp_ms", "price"]), 1.0)
    assert s.empty


def test_resample_last_price_forward_fills() -> None:
    df = pd.DataFrame(
        {
            "timestamp_ms": [0, 3000, 5000],
            "price": [100.0, 110.0, 105.0],
        }
    )
    s = _resample_last_price(df, 1.0)
    # 6 seconds of 1s grid with ffill.
    assert len(s) == 6
    assert s.iloc[0] == 100.0
    assert s.iloc[2] == 100.0  # ffilled
    assert s.iloc[3] == 110.0
    assert s.iloc[5] == 105.0


def test_cross_correlation_recovers_known_lag() -> None:
    rng = np.random.default_rng(0)
    n = 1000
    ra = rng.normal(0.0, 1.0, size=n)
    # rb at t == ra at (t - 3); positive lag => ra leads rb.
    rb = np.concatenate([np.zeros(3), ra[:-3]])

    lag_steps = np.arange(-5, 6)
    corrs = cross_correlation(ra, rb, lag_steps)
    best_lag = int(lag_steps[np.nanargmax(np.abs(corrs))])
    # In our convention lag>0 means rb shifted forward; the leader is ra.
    # rb[t] = ra[t-3]  ⇒  ra[t] = rb[t+3]  ⇒  corr(ra[t], rb[t+lag]) peaks at lag = -3? No:
    #   corr(ra[t], rb[t + lag]) = corr(ra[t], ra[t + lag - 3]) which peaks at lag = 3.
    assert best_lag == 3


def test_aligned_returns_handles_mismatched_indices() -> None:
    a = pd.Series(
        [100, 101, 102, 103],
        index=pd.to_datetime([1_000, 2_000, 3_000, 4_000], unit="ms", utc=True),
    )
    b = pd.Series(
        [200, 201, 202],
        index=pd.to_datetime([2_000, 3_000, 4_000], unit="ms", utc=True),
    )
    ra, rb = _aligned_returns(a, b)
    assert ra.shape == rb.shape
    assert ra.shape[0] == 2  # 3 aligned points -> 2 returns


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
    # Swap names so analyzer sees (kucoin, binance).
    result = analyze_pair("BTC/USDT", "kucoin", "binance", df_a, df_b, cfg)
    assert result is not None
    assert result.leader == "binance"
    assert result.follower == "kucoin"
    # When ex_a is the follower, best_lag_seconds is negative.
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


def test_analyze_all_handles_empty() -> None:
    cfg = _config_for_test()
    results = analyze_all(pd.DataFrame(), cfg)
    assert results == []
