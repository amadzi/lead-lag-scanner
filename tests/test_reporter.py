"""Tests for the reporter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lead_lag_scanner.analyzer import LeadLagResult
from lead_lag_scanner.config import ReportConfig
from lead_lag_scanner.reporter import (
    _gross_edge_bps,
    _net_edge_bps,
    _tradeability,
    write_reports,
)


def _make_result(
    *,
    symbol: str = "BTC/USDT",
    leader: str = "binance",
    follower: str = "kucoin",
    lag: float = 1.0,
    corr: float = 0.7,
    n_obs: int = 1000,
    flags: tuple[str, ...] = (),
) -> LeadLagResult:
    if leader == "none":
        exchange_a, exchange_b = "binance", "kucoin"
    else:
        exchange_a, exchange_b = (leader, follower) if lag > 0 else (follower, leader)
    return LeadLagResult(
        symbol=symbol,
        exchange_a=exchange_a,
        exchange_b=exchange_b,
        n_obs=n_obs,
        best_lag_seconds=lag,
        best_correlation=corr,
        correlation_at_zero=corr * 0.6,
        leader=leader,
        follower=follower,
        lag_ci_low_seconds=lag - 0.2,
        lag_ci_high_seconds=lag + 0.2,
        follower_return_std=0.001,
        flags=flags,
    )


def test_write_reports_filters_by_min_correlation(tmp_path: Path) -> None:
    cfg = ReportConfig(
        taker_bps=10.0,
        markdown_path=tmp_path / "r.md",
        json_path=tmp_path / "r.json",
        min_abs_correlation=0.5,
    )
    results = [
        _make_result(symbol="BTC/USDT", corr=0.9),
        _make_result(symbol="ETH/USDT", corr=0.3),
    ]
    md_path, json_path = write_reports(results, cfg)
    md = md_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert "BTC/USDT" in md
    assert "ETH/USDT" not in md
    assert len(payload["pairs"]) == 1
    assert payload["pairs"][0]["symbol"] == "BTC/USDT"


def test_write_reports_sorts_descending_when_costs_dominate(tmp_path: Path) -> None:
    """All three pairs are below cost; ranking is *least negative first*.

    Previously the reporter ranked by raw |corr|; we now rank by
    tradeability, which inverts the order in cost-dominated cases — the
    pair with the *lowest* edge wastage rises to the top, not the
    highest correlation. This test pins that behaviour so we don't
    accidentally regress to corr-only sorting.
    """

    cfg = ReportConfig(
        taker_bps=10.0,
        spread_bps=4.0,
        slippage_bps=2.0,
        markdown_path=tmp_path / "r.md",
        json_path=tmp_path / "r.json",
        min_abs_correlation=0.0,
    )
    results = [
        _make_result(symbol="A/USDT", corr=0.4),  # net most-positive (least negative)
        _make_result(symbol="B/USDT", corr=-0.95),
        _make_result(symbol="C/USDT", corr=0.7),
    ]
    md_path, _ = write_reports(results, cfg)
    md_lines = md_path.read_text(encoding="utf-8").splitlines()
    # First data row appears after the header rows.
    rows = [line for line in md_lines if line.startswith("| ") and "Symbol" not in line]
    rows = [r for r in rows if not r.startswith("|--")]
    symbols_in_order = [r.split("|")[1].strip() for r in rows]
    assert symbols_in_order == ["A/USDT", "C/USDT", "B/USDT"]


def test_write_reports_handles_no_results(tmp_path: Path) -> None:
    cfg = ReportConfig(
        taker_bps=10.0,
        markdown_path=tmp_path / "r.md",
        json_path=tmp_path / "r.json",
        min_abs_correlation=0.5,
    )
    md_path, json_path = write_reports([], cfg)
    md = md_path.read_text(encoding="utf-8")
    assert "No pairs met" in md
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["pairs"] == []


def test_edge_bps_includes_fee_offset(tmp_path: Path) -> None:
    cfg = ReportConfig(
        taker_bps=5.0,
        markdown_path=tmp_path / "r.md",
        json_path=tmp_path / "r.json",
        min_abs_correlation=0.0,
    )
    # |corr| * sigma * 1e4 = 0.5 * 0.001 * 1e4 = 5 bps; minus 2*5 = 10  => -5 bps.
    results = [_make_result(corr=0.5)]
    _, json_path = write_reports(results, cfg)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["pairs"][0]["edge_bps"] == -5.0


# _gross_edge_bps / _net_edge_bps / _tradeability ------------------------


def _cost_cfg(tmp_path: Path) -> ReportConfig:
    return ReportConfig(
        taker_bps=5.0,
        spread_bps=4.0,
        slippage_bps=2.0,
        markdown_path=tmp_path / "r.md",
        json_path=tmp_path / "r.json",
        min_abs_correlation=0.0,
    )


def test_gross_edge_bps_is_corr_times_sigma_times_1e4(tmp_path: Path) -> None:
    _ = _cost_cfg(tmp_path)
    # |0.7| * 0.001 * 1e4 = 7.0
    assert _gross_edge_bps(_make_result(corr=0.7)) == 7.0
    # Sign of correlation does not matter for the gross edge.
    assert _gross_edge_bps(_make_result(corr=-0.7)) == 7.0


def test_net_edge_bps_subtracts_round_trip_costs(tmp_path: Path) -> None:
    cfg = _cost_cfg(tmp_path)
    # gross = 7.0; net = 7.0 - 2*5.0 - 4.0 - 2.0 = -9.0
    r = _make_result(corr=0.7)
    assert _net_edge_bps(r, cfg) == -9.0


def test_net_edge_bps_positive_when_edge_dominates_costs(tmp_path: Path) -> None:
    cfg = _cost_cfg(tmp_path)
    r = LeadLagResult(
        symbol="BTC/USDT",
        exchange_a="binance",
        exchange_b="kucoin",
        n_obs=1000,
        best_lag_seconds=1.0,
        best_correlation=0.9,
        correlation_at_zero=0.5,
        leader="binance",
        follower="kucoin",
        lag_ci_low_seconds=0.8,
        lag_ci_high_seconds=1.2,
        follower_return_std=0.005,  # → gross = 0.9 * 0.005 * 1e4 = 45 bps
    )
    # 45 - (2*5 + 4 + 2) = 29 bps; allow rounding.
    assert _net_edge_bps(r, cfg) == pytest.approx(29.0)


def test_tradeability_zero_for_no_leader(tmp_path: Path) -> None:
    cfg = _cost_cfg(tmp_path)
    r = _make_result(leader="none", follower="none", lag=0.0, corr=0.99, n_obs=10_000)
    assert _tradeability(r, cfg) == 0.0


def test_tradeability_negative_when_costs_exceed_edge(tmp_path: Path) -> None:
    cfg = _cost_cfg(tmp_path)
    # gross = 7 bps, net = -9 bps; corr² = 0.49; sqrt(1000) ≈ 31.62
    r = _make_result(corr=0.7, n_obs=1000)
    score = _tradeability(r, cfg)
    assert score < 0
    assert score == pytest.approx(-9.0 * 0.49 * (1000**0.5))


def test_tradeability_ranks_more_observations_higher(tmp_path: Path) -> None:
    cfg = _cost_cfg(tmp_path)
    r_small = _make_result(corr=0.9, n_obs=200)
    r_large = _make_result(corr=0.9, n_obs=4000)
    # Same gross edge (9 bps), same corr, but sqrt(n) is 4.5x larger so score scales.
    assert _tradeability(r_large, cfg) < _tradeability(r_small, cfg) * 0.0  # both negative
    # Stronger statement: |large| > |small| because √4000 > √200.
    assert abs(_tradeability(r_large, cfg)) > abs(_tradeability(r_small, cfg))


def test_write_reports_sorts_by_tradeability(tmp_path: Path) -> None:
    cfg = _cost_cfg(tmp_path)
    # All three pass costs because we use sigma=0.005.
    results = [
        LeadLagResult(
            symbol=s,
            exchange_a="binance",
            exchange_b="kucoin",
            n_obs=n,
            best_lag_seconds=1.0,
            best_correlation=c,
            correlation_at_zero=c * 0.6,
            leader="binance",
            follower="kucoin",
            lag_ci_low_seconds=0.8,
            lag_ci_high_seconds=1.2,
            follower_return_std=0.005,
        )
        for s, c, n in [
            ("LOW/USDT", 0.5, 200),
            ("HIGH/USDT", 0.9, 4000),
            ("MID/USDT", 0.7, 1000),
        ]
    ]
    _, json_path = write_reports(results, cfg)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    symbols = [p["symbol"] for p in payload["pairs"]]
    assert symbols == ["HIGH/USDT", "MID/USDT", "LOW/USDT"]


def test_write_reports_emits_flags_in_payload(tmp_path: Path) -> None:
    cfg = _cost_cfg(tmp_path)
    r = _make_result(corr=0.95, n_obs=80, flags=("low_n_high_corr",))
    _, json_path = write_reports([r], cfg)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["pairs"][0]["flags"] == ["low_n_high_corr"]
    assert payload["spread_bps_assumption"] == 4.0
    assert payload["slippage_bps_assumption"] == 2.0
    md = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "low_n_high_corr" in md
