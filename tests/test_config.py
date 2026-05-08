"""Tests for the YAML config loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lead_lag_scanner.config import load_config


def test_load_default_config() -> None:
    cfg = load_config()
    assert len(cfg.exchanges) > 0
    assert len(cfg.symbols) > 0
    assert cfg.collector.duration > 0
    assert cfg.analyzer.lag_grid.start <= 0 <= cfg.analyzer.lag_grid.stop
    assert cfg.analyzer.lag_grid.step > 0


def test_load_overrides(tmp_path: Path) -> None:
    payload = {
        "exchanges": ["binance", "okx"],
        "symbols": ["BTC/USDT"],
        "collector": {"duration": 10},
        "analyzer": {"min_obs": 1, "lag_grid": {"start": -1.0, "stop": 1.0, "step": 0.5}},
        "report": {"taker_bps": 5.0, "min_abs_correlation": 0.1},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cfg = load_config(cfg_path)
    assert cfg.exchanges == ("binance", "okx")
    assert cfg.symbols == ("BTC/USDT",)
    assert cfg.collector.duration == 10
    assert cfg.analyzer.min_obs == 1
    assert cfg.analyzer.lag_grid.start == -1.0
    assert cfg.analyzer.lag_grid.stop == 1.0
    assert cfg.analyzer.lag_grid.step == 0.5
    assert cfg.report.taker_bps == 5.0


def test_missing_exchanges_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({"symbols": ["BTC/USDT"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="exchanges"):
        load_config(cfg_path)


def test_missing_symbols_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({"exchanges": ["binance"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="symbols"):
        load_config(cfg_path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")
