"""Typed configuration loader.

The config is intentionally a small, frozen dataclass tree so downstream code
can pass it around without worrying about mutation, and so type checkers can
catch typos in field names.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
    """Knobs that control the lead-lag analyzer.

    The default behaviour applies four data-quality fixes that matter when
    scanning many exchanges at once:

    * **Clock-skew calibration** (``clock_skew_calibration``): each exchange's
      timestamps are re-centered using the median of
      ``local_recv_ts_ns − exchange_ts_ms`` per exchange, so that systematic
      offsets between exchange matching-engine clocks (often ±100–500 ms,
      sometimes seconds) do not show up as spurious lag.
    * **Sanity filter** (``max_clock_drift_seconds``): trades whose
      ``|local_recv − exchange_ts|`` exceeds this many seconds are dropped as
      probable API quirks (e.g. an exchange returning microseconds in a field
      documented as milliseconds).
    * **Active-bar masking** (``active_mask``): the cross-correlation only
      uses bars where *both* exchanges actually saw a trade in the bar. This
      prevents the forward-fill of stale prices on low-tick-frequency
      exchanges from creating phantom lag.
    * **Low-tick filter** (``min_active_rate``): an (exchange, symbol) is
      dropped from the analysis if the fraction of bars containing a real
      trade is below this threshold. The ratio is computed against the bar
      count of the *busier* side so a slow venue cannot anchor a fast one.

    Setting any of the toggles to ``False`` (or the threshold to 0) reverts
    to the legacy un-masked / un-calibrated behaviour, which is useful for
    sanity-checking the impact of each fix.
    """

    resample_seconds: float = 1.0
    lag_grid: LagGrid = field(default_factory=LagGrid)
    min_obs: int = 600
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    clock_skew_calibration: bool = True
    max_clock_drift_seconds: float = 3600.0
    active_mask: bool = True
    min_active_rate: float = 0.05


@dataclass(frozen=True, slots=True)
class ReportConfig:
    """Cost assumptions and report output paths.

    The ``edge_bps`` proxy in :func:`reporter._edge_bps` only nets out
    ``2 × taker_bps`` (one side enters, one side exits). Spread and
    slippage are *not* free and are typically larger than the taker fee on
    the kind of microcap pairs that surface in this scanner. ``spread_bps``
    and ``slippage_bps`` are subtracted in :func:`reporter._net_edge_bps`
    so the dashboard can show a realistic net-of-cost number alongside the
    naive proxy. Both are round-trip totals (entry + exit), in basis
    points, and configurable per-deployment.
    """

    taker_bps: float = 10.0
    spread_bps: float = 4.0
    slippage_bps: float = 2.0
    markdown_path: Path = Path("reports/report.md")
    json_path: Path = Path("reports/leaders.json")
    min_abs_correlation: float = 0.4


@dataclass(frozen=True, slots=True)
class DashboardConfig:
    """Live terminal dashboard knobs.

    The dashboard re-reads the parquet shards and re-runs the analyzer on a
    fixed schedule, then renders the top-K pairs sorted by ``sort_by``.
    Defaults are deliberately *looser* than :class:`ReportConfig` /
    :class:`AnalyzerConfig` so the dashboard shows preliminary signal a few
    minutes after collection starts (instead of waiting for ``min_obs=600``).

    ``hide_flags`` is a tuple of sanity-flag names to suppress in the
    rendered table. The default hides ``boundary_lag`` and ``wide_ci`` —
    pairs whose lag is pinned to the grid edge or whose CI spans both
    signs almost never trade out. Override with an empty tuple to see
    everything.
    """

    refresh_seconds: float = 30.0
    top_k: int = 30
    sort_by: str = "tradeability"
    min_abs_correlation: float = 0.3
    min_obs: int = 60
    bootstrap_n_iter: int = 30
    only_directional: bool = False  # if True, drop pairs with leader == "none"
    hide_flags: tuple[str, ...] = ("boundary_lag", "wide_ci")


@dataclass(frozen=True, slots=True)
class Config:
    exchanges: tuple[str, ...]
    symbols: tuple[str, ...]
    collector: CollectorConfig = field(default_factory=CollectorConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    analyzer: AnalyzerConfig = field(default_factory=AnalyzerConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)


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
        clock_skew_calibration=bool(d.get("clock_skew_calibration", True)),
        max_clock_drift_seconds=float(d.get("max_clock_drift_seconds", 3600.0)),
        active_mask=bool(d.get("active_mask", True)),
        min_active_rate=float(d.get("min_active_rate", 0.05)),
    )


def _load_report(d: dict[str, Any]) -> ReportConfig:
    return ReportConfig(
        taker_bps=float(d.get("taker_bps", 10.0)),
        spread_bps=float(d.get("spread_bps", 4.0)),
        slippage_bps=float(d.get("slippage_bps", 2.0)),
        markdown_path=_coerce_path(d.get("markdown_path"), Path("reports/report.md")),
        json_path=_coerce_path(d.get("json_path"), Path("reports/leaders.json")),
        min_abs_correlation=float(d.get("min_abs_correlation", 0.4)),
    )


_DASHBOARD_SORT_KEYS = (
    "tradeability",
    "edge_bps",
    "net_edge_bps",
    "abs_corr",
    "n_obs",
    "lag_abs",
)
_KNOWN_FLAGS = ("boundary_lag", "wide_ci", "low_n_high_corr", "zero_lag")


def _coerce_flag_tuple(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    """Parse ``hide_flags`` from YAML (list / CSV string / None)."""

    if value is None:
        return default
    if isinstance(value, str):
        items = [s.strip() for s in value.split(",") if s.strip()]
    elif isinstance(value, (list, tuple)):
        items = [str(s).strip() for s in value if str(s).strip()]
    else:
        raise ValueError(f"dashboard.hide_flags must be list or CSV string, got {value!r}")
    bad = [f for f in items if f not in _KNOWN_FLAGS]
    if bad:
        raise ValueError(
            f"dashboard.hide_flags has unknown flag(s): {bad!r}; known: {_KNOWN_FLAGS}"
        )
    seen: set[str] = set()
    out: list[str] = []
    for f in items:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return tuple(out)


def _load_dashboard(d: dict[str, Any]) -> DashboardConfig:
    sort_by = str(d.get("sort_by", "tradeability"))
    if sort_by not in _DASHBOARD_SORT_KEYS:
        raise ValueError(
            f"dashboard.sort_by must be one of {_DASHBOARD_SORT_KEYS}, got {sort_by!r}"
        )
    return DashboardConfig(
        refresh_seconds=float(d.get("refresh_seconds", 30.0)),
        top_k=int(d.get("top_k", 30)),
        sort_by=sort_by,
        min_abs_correlation=float(d.get("min_abs_correlation", 0.3)),
        min_obs=int(d.get("min_obs", 60)),
        bootstrap_n_iter=int(d.get("bootstrap_n_iter", 30)),
        only_directional=bool(d.get("only_directional", False)),
        hide_flags=_coerce_flag_tuple(d.get("hide_flags"), ("boundary_lag", "wide_ci")),
    )


RUNTIME_SYMBOLS_FILENAME = "runtime-symbols.yaml"


def runtime_symbols_path(data_dir: Path) -> Path:
    """Return the path where user-added tickers are persisted."""

    return Path(data_dir) / RUNTIME_SYMBOLS_FILENAME


def load_runtime_symbols(data_dir: Path) -> tuple[str, ...]:
    """Load extra tickers added through the web dashboard.

    The file is a tiny YAML doc with a single ``symbols:`` list, e.g.::

        symbols:
          - PEPE/USDT
          - WIF/USDT

    Missing file ⇒ empty tuple. Malformed file ⇒ empty tuple (we never want
    a stray runtime symbol file to break the collector).
    """

    path = runtime_symbols_path(data_dir)
    if not path.exists():
        return ()
    try:
        with path.open("r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError):
        return ()
    symbols = raw.get("symbols") or []
    if not isinstance(symbols, list):
        return ()
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in symbols:
        s = str(item).strip()
        if s and s not in seen:
            cleaned.append(s)
            seen.add(s)
    return tuple(cleaned)


def save_runtime_symbols(data_dir: Path, symbols: tuple[str, ...]) -> Path:
    """Atomically persist the runtime ticker list to disk."""

    path = runtime_symbols_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"symbols": list(symbols)}
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    tmp.replace(path)
    return path


def merge_runtime_symbols(config: Config) -> Config:
    """Return a copy of ``config`` with ``runtime-symbols.yaml`` appended.

    Order: configured symbols first (preserves the curated coverage-sorted
    order), then runtime additions in insertion order, deduplicated.
    """

    extras = load_runtime_symbols(config.storage.data_dir)
    if not extras:
        return config
    seen: set[str] = set(config.symbols)
    merged: list[str] = list(config.symbols)
    for s in extras:
        if s not in seen:
            merged.append(s)
            seen.add(s)
    if len(merged) == len(config.symbols):
        return config
    return replace(config, symbols=tuple(merged))


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
        dashboard=_load_dashboard(raw.get("dashboard") or {}),
    )
