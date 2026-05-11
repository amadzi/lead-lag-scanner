"""Tests for the live dashboard rendering and filtering helpers."""

from __future__ import annotations

from lead_lag_scanner.analyzer import LeadLagResult
from lead_lag_scanner.config import (
    AnalyzerConfig,
    BootstrapConfig,
    DashboardConfig,
    LagGrid,
    ReportConfig,
)
from lead_lag_scanner.dashboard import (
    _effective_analyzer_config,
    _filter_and_sort,
    _sort_key,
    render_table,
)


def _result(
    *,
    symbol: str = "BTC/USDT",
    leader: str = "binance",
    follower: str = "kucoin",
    lag: float = 1.0,
    corr: float = 0.7,
    sigma: float = 0.001,
    n_obs: int = 1000,
) -> LeadLagResult:
    if lag > 0:
        ex_a, ex_b = leader, follower
    elif lag < 0:
        ex_a, ex_b = follower, leader
    else:
        ex_a, ex_b = leader, follower
    return LeadLagResult(
        symbol=symbol,
        exchange_a=ex_a,
        exchange_b=ex_b,
        n_obs=n_obs,
        best_lag_seconds=lag,
        best_correlation=corr,
        correlation_at_zero=corr * 0.6,
        leader=leader if lag != 0 else "none",
        follower=follower if lag != 0 else "none",
        lag_ci_low_seconds=lag - 0.2,
        lag_ci_high_seconds=lag + 0.2,
        follower_return_std=sigma,
    )


def _report_cfg() -> ReportConfig:
    return ReportConfig(
        taker_bps=10.0,
        spread_bps=4.0,
        slippage_bps=2.0,
        min_abs_correlation=0.0,
    )


def _dash(**overrides: object) -> DashboardConfig:
    base = {
        "refresh_seconds": 30.0,
        "top_k": 30,
        "sort_by": "edge_bps",
        "min_abs_correlation": 0.3,
        "min_obs": 60,
        "bootstrap_n_iter": 30,
        "only_directional": False,
        "hide_flags": (),
    }
    base.update(overrides)
    return DashboardConfig(**base)  # type: ignore[arg-type]


def test_sort_key_edge_bps_orders_by_descending_edge() -> None:
    # |corr|*sigma*1e4 - 2*taker_bps;  taker_bps=10
    # high: 0.9 * 0.005 * 1e4 - 20 = 25
    # low:  0.5 * 0.001 * 1e4 - 20 = -15
    high = _result(symbol="HIGH", corr=0.9, sigma=0.005)
    low = _result(symbol="LOW", corr=0.5, sigma=0.001)
    out = sorted([low, high], key=_sort_key("edge_bps", _report_cfg()))
    assert out[0] is high
    assert out[1] is low


def test_sort_key_other_axes() -> None:
    a = _result(symbol="A", corr=0.4, lag=0.5, n_obs=200)
    b = _result(symbol="B", corr=0.9, lag=2.5, n_obs=100)
    cfg = _report_cfg()
    out_corr = sorted([a, b], key=_sort_key("abs_corr", cfg))
    assert out_corr[0] is b
    out_lag = sorted([a, b], key=_sort_key("lag_abs", cfg))
    assert out_lag[0] is b
    out_obs = sorted([a, b], key=_sort_key("n_obs", cfg))
    assert out_obs[0] is a


def test_sort_key_tradeability_orders_by_descending_score() -> None:
    cfg = _report_cfg()
    # tradeability = (gross - 2*taker - spread - slippage) * corr² * √n
    # high: (45 - 20 - 4 - 2) * 0.81 * √1000 ≈ 487
    # low:  (5  - 20 - 4 - 2) * 0.25 * √1000 ≈ -166
    high = _result(symbol="HIGH", corr=0.9, sigma=0.005, n_obs=1000)
    low = _result(symbol="LOW", corr=0.5, sigma=0.001, n_obs=1000)
    out = sorted([low, high], key=_sort_key("tradeability", cfg))
    assert out[0] is high


def test_sort_key_net_edge_orders_by_descending_net() -> None:
    cfg = _report_cfg()
    high = _result(symbol="HIGH", corr=0.9, sigma=0.005)
    low = _result(symbol="LOW", corr=0.5, sigma=0.001)
    out = sorted([low, high], key=_sort_key("net_edge_bps", cfg))
    assert out[0] is high


def test_filter_and_sort_drops_pairs_with_hidden_flags() -> None:
    cfg = _report_cfg()
    keep = _result(symbol="KEEP", corr=0.5)
    flagged = LeadLagResult(
        symbol="FLAG",
        exchange_a="binance",
        exchange_b="kucoin",
        n_obs=1000,
        best_lag_seconds=2.0,
        best_correlation=0.5,
        correlation_at_zero=0.3,
        leader="binance",
        follower="kucoin",
        lag_ci_low_seconds=1.8,
        lag_ci_high_seconds=2.2,
        follower_return_std=0.001,
        flags=("boundary_lag",),
    )
    out = _filter_and_sort([keep, flagged], _dash(hide_flags=("boundary_lag",)), cfg)
    assert [r.symbol for r in out] == ["KEEP"]


def test_filter_drops_low_correlation() -> None:
    keep = _result(symbol="K", corr=0.5)
    drop = _result(symbol="D", corr=0.1)
    out = _filter_and_sort([keep, drop], _dash(min_abs_correlation=0.3), _report_cfg())
    assert [r.symbol for r in out] == ["K"]


def test_filter_only_directional_drops_none_leader() -> None:
    direct = _result(symbol="D", lag=1.0, corr=0.5)
    flat = _result(symbol="F", lag=0.0, corr=0.9)
    out = _filter_and_sort(
        [direct, flat], _dash(only_directional=True, min_abs_correlation=0.0), _report_cfg()
    )
    assert [r.symbol for r in out] == ["D"]


def test_render_table_includes_top_k_rows_in_sorted_order() -> None:
    a = _result(symbol="A/USDT", corr=0.4, sigma=0.001)  # edge =  4 - 20 = -16
    b = _result(symbol="B/USDT", corr=0.9, sigma=0.005)  # edge = 45 - 20 = +25
    c = _result(symbol="C/USDT", corr=0.6, sigma=0.002)  # edge = 12 - 20 = -8
    table = render_table([a, b, c], _report_cfg(), _dash(top_k=2, min_abs_correlation=0.0))
    assert table.row_count == 2
    cells = [str(table.columns[0]._cells[i]) for i in range(table.row_count)]
    assert cells == ["B/USDT", "C/USDT"]


def test_render_table_handles_empty_results() -> None:
    table = render_table([], _report_cfg(), _dash())
    assert table.row_count == 0
    assert "lead-lag-scanner" in str(table.title)
    assert "trades=" not in (table.caption or "")


def test_render_table_caption_includes_metadata() -> None:
    table = render_table(
        [_result()],
        _report_cfg(),
        _dash(),
        n_trades=12345,
        n_exchanges=20,
        n_symbols=10,
        cycle_seconds=2.5,
    )
    caption = str(table.caption)
    assert "trades=12,345" in caption
    assert "exchanges=20" in caption
    assert "symbols=10" in caption
    assert "cycle=2.5s" in caption


def test_effective_analyzer_config_overrides_min_obs_and_bootstrap() -> None:
    base = AnalyzerConfig(
        resample_seconds=1.0,
        lag_grid=LagGrid(),
        min_obs=600,
        bootstrap=BootstrapConfig(block_size=30, n_iter=200),
    )
    eff = _effective_analyzer_config(base, _dash(min_obs=42, bootstrap_n_iter=7))
    assert eff.min_obs == 42
    assert eff.bootstrap.n_iter == 7
    assert eff.bootstrap.block_size == 30
    assert eff.resample_seconds == 1.0
