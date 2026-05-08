"""Tests for the reporter."""

from __future__ import annotations

import json
from pathlib import Path

from lead_lag_scanner.analyzer import LeadLagResult
from lead_lag_scanner.config import ReportConfig
from lead_lag_scanner.reporter import write_reports


def _make_result(
    *,
    symbol: str = "BTC/USDT",
    leader: str = "binance",
    follower: str = "kucoin",
    lag: float = 1.0,
    corr: float = 0.7,
    n_obs: int = 1000,
) -> LeadLagResult:
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


def test_write_reports_sorts_by_abs_correlation(tmp_path: Path) -> None:
    cfg = ReportConfig(
        taker_bps=10.0,
        markdown_path=tmp_path / "r.md",
        json_path=tmp_path / "r.json",
        min_abs_correlation=0.0,
    )
    results = [
        _make_result(symbol="A/USDT", corr=0.4),
        _make_result(symbol="B/USDT", corr=-0.95),
        _make_result(symbol="C/USDT", corr=0.7),
    ]
    md_path, _ = write_reports(results, cfg)
    md_lines = md_path.read_text(encoding="utf-8").splitlines()
    # First data row appears after the header rows.
    rows = [line for line in md_lines if line.startswith("| ") and "Symbol" not in line]
    rows = [r for r in rows if not r.startswith("|--")]
    symbols_in_order = [r.split("|")[1].strip() for r in rows]
    assert symbols_in_order == ["B/USDT", "C/USDT", "A/USDT"]


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
