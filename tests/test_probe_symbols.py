"""Tests for the symbol-probe helpers and ``probe-symbols`` CLI command.

We never hit the real ccxt clients here — :func:`probe_usdt_symbols`
delegates to :func:`_list_quoted_spot_symbols`, which we monkeypatch.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from click.testing import CliRunner

from lead_lag_scanner import cli as cli_mod
from lead_lag_scanner import exchanges as ex_mod
from lead_lag_scanner.config import (
    AnalyzerConfig,
    BootstrapConfig,
    CollectorConfig,
    Config,
    DashboardConfig,
    LagGrid,
    ReportConfig,
    StorageConfig,
)
from lead_lag_scanner.exchanges import probe_usdt_symbols


@pytest.fixture
def fake_listings(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """Stub out per-exchange symbol discovery with deterministic data."""

    listings: dict[str, list[str]] = {
        "binance": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "PEPE/USDT"],
        "okx": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "WIF/USDT"],
        "kucoin": ["BTC/USDT", "ETH/USDT", "DOGE/USDT", "PEPE/USDT", "WIF/USDT"],
        "bybit": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        "gate": ["BTC/USDT", "ETH/USDT", "MEME/USDT"],
        "deadex": [],
    }

    async def _fake_list(exchange_id: str, quote: str) -> list[str]:
        assert quote == "USDT"
        return list(listings.get(exchange_id, []))

    monkeypatch.setattr(ex_mod, "_list_quoted_spot_symbols", _fake_list)
    return listings


def test_probe_usdt_symbols_aggregates_coverage(fake_listings: dict[str, list[str]]) -> None:
    coverage = asyncio.run(probe_usdt_symbols(list(fake_listings.keys())))

    # Every symbol that appears anywhere makes it into the coverage map.
    expected_syms = {sym for syms in fake_listings.values() for sym in syms}
    assert set(coverage.keys()) == expected_syms

    # BTC/USDT is on every non-empty exchange.
    assert set(coverage["BTC/USDT"]) == {"binance", "okx", "kucoin", "bybit", "gate"}
    # Single-listed symbol stays single-listed.
    assert coverage["MEME/USDT"] == ["gate"]
    # Order preserved — exchanges are appended in input order.
    assert coverage["BTC/USDT"][0] == "binance"


def test_probe_usdt_symbols_skips_empty_exchanges(
    fake_listings: dict[str, list[str]],
) -> None:
    coverage = asyncio.run(probe_usdt_symbols(["deadex", "binance"]))
    # deadex returned [] so it doesn't appear in any coverage list.
    for exchanges in coverage.values():
        assert "deadex" not in exchanges
    assert "BTC/USDT" in coverage


def test_probe_symbols_cli_filters_and_writes_yaml(
    tmp_path: Path,
    fake_listings: dict[str, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: ``probe-symbols`` honours --min-exchanges, --top, --out."""

    # Build a tiny config that uses our fake exchanges only. We override
    # the config loader so we don't need a YAML file on disk.
    fake_cfg = Config(
        exchanges=tuple(fake_listings.keys()),
        symbols=("BTC/USDT",),
        collector=CollectorConfig(
            duration=1.0,
            rate_limit_safety=1.0,
            prefer_websocket=False,
            rest_poll_interval=1.0,
        ),
        storage=StorageConfig(data_dir=tmp_path / "data", duckdb_path=tmp_path / "x.duckdb"),
        analyzer=AnalyzerConfig(
            resample_seconds=1.0,
            lag_grid=LagGrid(start=-1.0, stop=1.0, step=1.0),
            min_obs=1,
            bootstrap=BootstrapConfig(block_size=2, n_iter=2),
        ),
        report=ReportConfig(
            taker_bps=10.0,
            markdown_path=tmp_path / "r.md",
            json_path=tmp_path / "r.json",
            min_abs_correlation=0.0,
        ),
        dashboard=DashboardConfig(
            refresh_seconds=1.0,
            top_k=1,
            sort_by="edge_bps",
            min_abs_correlation=0.0,
            min_obs=1,
            bootstrap_n_iter=1,
            only_directional=False,
        ),
    )
    monkeypatch.setattr(cli_mod, "load_config", lambda _path: fake_cfg)

    out_path = tmp_path / "snippet.yaml"
    result = CliRunner().invoke(
        cli_mod.cli,
        [
            "probe-symbols",
            "--min-exchanges",
            "3",
            "--top",
            "10",
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output

    snippet = out_path.read_text()
    # >=3 listings: BTC, ETH, DOGE on >=3 exchanges; SOL on 3 (binance, okx, bybit).
    # PEPE on 2, WIF on 2, MEME on 1 — must NOT appear.
    assert "BTC/USDT" in snippet
    assert "ETH/USDT" in snippet
    assert "DOGE/USDT" in snippet
    assert "SOL/USDT" in snippet
    assert "PEPE/USDT" not in snippet
    assert "WIF/USDT" not in snippet
    assert "MEME/USDT" not in snippet
    # Snippet starts with a YAML-friendly header.
    assert snippet.lstrip().startswith("# probe-symbols")
    assert "symbols:" in snippet
