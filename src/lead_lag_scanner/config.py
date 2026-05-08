"""Typed configuration loader.

The config is intentionally a small, frozen dataclass tree so downstream code
can pass it around without worrying about mutation, and so type checkers can
catch typos in field names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


@dataclass(frozen=True, slots=True)
class CollectorConfig:
    duration: float = 3600.0
    rate_limit_safety: float = 1.2
    prefer_websocket: bool = True
    rest_poll_interval: float = 1.0


@dataclass(frozen=True, slots=True)
class StorageConfig:
    data_dir: Path = Path("data")
    duckdb_path: Path = Path("data/trades.duckdb")


@dataclass(frozen=True, slots=True)
class LagGrid:
    start: float = -5.0
    stop: float = 5.0
    step: float = 0.1


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    block_size: int = 30
    n_iter: int = 200


@dataclass(frozen=True, slots=True)
class AnalyzerConfig:
    resample_seconds: float = 1.0
    lag_grid: LagGrid = field(default_factory=LagGrid)
    min_obs: int = 600
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)


@dataclass(frozen=True, slots=True)
class ReportConfig:
    taker_bps: float = 10.0
    markdown_path: Path = Path("reports/report.md")
    json_path: Path = Path("reports/leaders.json")
    min_abs_correlation: float = 0.4


@dataclass(frozen=True, slots=True)
class Config:
    exchanges: tuple[str, ...]
    symbols: tuple[str, ...]
    collector: CollectorConfig = field(default_factory=CollectorConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    analyzer: AnalyzerConfig = field(default_factory=AnalyzerConfig)
    report: ReportConfig = field(default_factory=ReportConfig)


def _coerce_path(value: Any, default: Path) -> Path:
    if value is None:
        return default
    return Path(value)


def _load_collector(d: dict[str, Any]) -> CollectorConfig:
    return CollectorConfig(
        duration=float(d.get("duration", 3600.0)),
        rate_limit_safety=float(d.get("rate_limit_safety", 1.2)),
        prefer_websocket=bool(d.get("prefer_websocket", True)),
        rest_poll_interval=float(d.get("rest_poll_interval", 1.0)),
    )


def _load_storage(d: dict[str, Any]) -> StorageConfig:
    return StorageConfig(
        data_dir=_coerce_path(d.get("data_dir"), Path("data")),
        duckdb_path=_coerce_path(d.get("duckdb_path"), Path("data/trades.duckdb")),
    )


def _load_analyzer(d: dict[str, Any]) -> AnalyzerConfig:
    lg = d.get("lag_grid", {}) or {}
    bs = d.get("bootstrap", {}) or {}
    return AnalyzerConfig(
        resample_seconds=float(d.get("resample_seconds", 1.0)),
        lag_grid=LagGrid(
            start=float(lg.get("start", -5.0)),
            stop=float(lg.get("stop", 5.0)),
            step=float(lg.get("step", 0.1)),
        ),
        min_obs=int(d.get("min_obs", 600)),
        bootstrap=BootstrapConfig(
            block_size=int(bs.get("block_size", 30)),
            n_iter=int(bs.get("n_iter", 200)),
        ),
    )


def _load_report(d: dict[str, Any]) -> ReportConfig:
    return ReportConfig(
        taker_bps=float(d.get("taker_bps", 10.0)),
        markdown_path=_coerce_path(d.get("markdown_path"), Path("reports/report.md")),
        json_path=_coerce_path(d.get("json_path"), Path("reports/leaders.json")),
        min_abs_correlation=float(d.get("min_abs_correlation", 0.4)),
    )


def load_config(path: Path | str | None = None) -> Config:
    """Load a :class:`Config` from a YAML file.

    If ``path`` is ``None`` the bundled ``config/default.yaml`` is used.
    Unknown keys are ignored — the contract is forward-compatible.
    """

    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    exchanges_raw = raw.get("exchanges") or []
    symbols_raw = raw.get("symbols") or []
    if not exchanges_raw:
        raise ValueError(f"`exchanges` must be a non-empty list in {cfg_path}")
    if not symbols_raw:
        raise ValueError(f"`symbols` must be a non-empty list in {cfg_path}")

    return Config(
        exchanges=tuple(str(x) for x in exchanges_raw),
        symbols=tuple(str(x) for x in symbols_raw),
        collector=_load_collector(raw.get("collector") or {}),
        storage=_load_storage(raw.get("storage") or {}),
        analyzer=_load_analyzer(raw.get("analyzer") or {}),
        report=_load_report(raw.get("report") or {}),
    )
