"""HTTP backend for the in-browser lead-lag dashboard.

This module exposes a small FastAPI application that re-uses the existing
analyzer to surface the latest lead-lag results in a friendly web UI. It is
intentionally read-mostly:

* a background refresh task re-loads the parquet shards and re-runs the
  analyzer every ``dashboard.refresh_seconds`` seconds, populating an
  in-memory snapshot;
* HTTP endpoints just read from that snapshot;
* the only mutating endpoints manage the runtime-symbols YAML file (a
  user-curated list of extra tickers to collect).

The frontend (single-file HTML/JS, see ``frontend/index.html``) renders
everything in Russian and offers filters / sort / ticker management.
"""

from __future__ import annotations

import re
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .analyzer import (
    ExchangeDiagnostics,
    LeadLagResult,
    analyze_all_with_diagnostics,
)
from .config import (
    AnalyzerConfig,
    BootstrapConfig,
    Config,
    DashboardConfig,
    load_runtime_symbols,
    merge_runtime_symbols,
    save_runtime_symbols,
)
from .reporter import _edge_bps
from .storage import load_trades

log = structlog.get_logger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
INDEX_HTML_PATH = FRONTEND_DIR / "index.html"

# A ccxt-unified symbol is ``BASE/QUOTE`` with both sides being short
# alpha-numeric tokens (and optionally a ``-`` separator inside the base, e.g.
# ``1000PEPE``). We deliberately keep this strict so we don't accept
# accidental whitespace / control characters from the UI.
_SYMBOL_RE = re.compile(r"^[A-Z0-9]{1,15}(?:[-_][A-Z0-9]{1,5})?/[A-Z0-9]{2,8}$")


@dataclass(slots=True)
class _Snapshot:
    """Latest analysis results plus the metadata to render them."""

    computed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    n_trades: int = 0
    exchanges_with_trades: tuple[str, ...] = ()
    symbols_with_trades: tuple[str, ...] = ()
    results: list[LeadLagResult] = field(default_factory=list)
    diagnostics: list[ExchangeDiagnostics] = field(default_factory=list)
    last_error: str | None = None
    last_compute_seconds: float = 0.0


@dataclass(slots=True)
class WebState:
    """Mutable, lock-protected web-server state.

    The background refresh task writes here; HTTP handlers read with the
    lock held briefly. ``config`` is rebuilt at each refresh so a freshly
    added runtime ticker shows up in the *next* snapshot.
    """

    base_config: Config
    snapshot: _Snapshot = field(default_factory=_Snapshot)
    lock: threading.Lock = field(default_factory=threading.Lock)
    refresh_seconds: float = 30.0


def _effective_analyzer_config(base: AnalyzerConfig, dash: DashboardConfig) -> AnalyzerConfig:
    """Lower ``min_obs`` / bootstrap iters so the web view is responsive."""

    new_bootstrap = BootstrapConfig(
        block_size=base.bootstrap.block_size,
        n_iter=dash.bootstrap_n_iter,
    )
    return replace(base, min_obs=dash.min_obs, bootstrap=new_bootstrap)


def _refresh_snapshot(state: WebState) -> _Snapshot:
    """Synchronously reload trades and re-run the analyzer."""

    started = time.monotonic()
    config = merge_runtime_symbols(state.base_config)
    try:
        trades = load_trades(config.storage.data_dir)
    except Exception as exc:
        # ``exc_info=True`` so structlog prints the full traceback, not
        # just the message — invaluable when the error is something like
        # ``operands could not be broadcast together`` that is meaningful
        # only with line numbers.
        log.warning("web: load_trades failed", error=str(exc), exc_info=True)
        return _Snapshot(last_error=f"load_trades: {exc!s}")
    if trades.empty:
        return _Snapshot(last_error=None)
    eff_analyzer = _effective_analyzer_config(config.analyzer, config.dashboard)
    try:
        out = analyze_all_with_diagnostics(trades, eff_analyzer)
    except Exception as exc:
        log.warning("web: analyze_all failed", error=str(exc), exc_info=True)
        return _Snapshot(
            n_trades=len(trades),
            exchanges_with_trades=tuple(sorted(trades["exchange"].unique().tolist())),
            symbols_with_trades=tuple(sorted(trades["symbol"].unique().tolist())),
            last_error=f"analyze: {exc!s}",
        )
    elapsed = time.monotonic() - started
    return _Snapshot(
        computed_at=datetime.now(UTC),
        n_trades=len(trades),
        exchanges_with_trades=tuple(sorted(trades["exchange"].unique().tolist())),
        symbols_with_trades=tuple(sorted(trades["symbol"].unique().tolist())),
        results=out.results,
        diagnostics=out.diagnostics,
        last_compute_seconds=elapsed,
    )


def _refresh_loop(state: WebState, stop_event: threading.Event) -> None:
    """Refresh the snapshot every ``refresh_seconds`` until ``stop_event``."""

    while not stop_event.is_set():
        snap = _refresh_snapshot(state)
        with state.lock:
            state.snapshot = snap
        stop_event.wait(state.refresh_seconds)


# ---------------------------------------------------------------------------
# Pydantic schemas (JSON serialisation contract)
# ---------------------------------------------------------------------------


class PairOut(BaseModel):
    symbol: str
    exchange_a: str
    exchange_b: str
    leader: str
    follower: str
    best_lag_seconds: float
    best_correlation: float
    abs_correlation: float
    correlation_at_zero: float
    lag_ci_low_seconds: float
    lag_ci_high_seconds: float
    follower_return_std: float
    edge_bps: float
    n_obs: int
    n_co_active: int
    active_rate_a: float
    active_rate_b: float


class DiagnosticOut(BaseModel):
    exchange: str
    n_trades: int
    transport_latency_p50_ms: float
    clock_offset_ms: float


class StateOut(BaseModel):
    computed_at: str
    refresh_seconds: float
    last_compute_seconds: float
    last_error: str | None
    n_trades: int
    n_pairs: int
    exchanges_configured: list[str]
    exchanges_with_trades: list[str]
    symbols_configured: list[str]
    symbols_with_trades: list[str]
    runtime_symbols: list[str]
    taker_bps: float


class PairsOut(BaseModel):
    computed_at: str
    n_total: int
    n_filtered: int
    pairs: list[PairOut]
    diagnostics: list[DiagnosticOut]


class AddSymbolIn(BaseModel):
    symbol: str = Field(..., min_length=3, max_length=32)


class SymbolsOut(BaseModel):
    runtime_symbols: list[str]
    message: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_symbol(raw: str) -> str:
    """Canonicalise user input: trim, upper-case, validate format."""

    clean = raw.strip().upper().replace(" ", "")
    if not _SYMBOL_RE.match(clean):
        raise HTTPException(
            status_code=400,
            detail=(
                "Тикер должен быть в формате BASE/QUOTE латиницей и цифрами "
                "(например, BTC/USDT, 1000PEPE/USDT, KAS-PERP/USDT)."
            ),
        )
    return clean


def _result_to_pair(r: LeadLagResult, taker_bps: float) -> PairOut:
    return PairOut(
        symbol=r.symbol,
        exchange_a=r.exchange_a,
        exchange_b=r.exchange_b,
        leader=r.leader,
        follower=r.follower,
        best_lag_seconds=float(r.best_lag_seconds),
        best_correlation=float(r.best_correlation),
        abs_correlation=abs(float(r.best_correlation)),
        correlation_at_zero=float(r.correlation_at_zero),
        lag_ci_low_seconds=float(r.lag_ci_low_seconds),
        lag_ci_high_seconds=float(r.lag_ci_high_seconds),
        follower_return_std=float(r.follower_return_std),
        edge_bps=float(_edge_bps(r, taker_bps)),
        n_obs=int(r.n_obs),
        n_co_active=int(r.n_co_active),
        active_rate_a=float(r.active_rate_a),
        active_rate_b=float(r.active_rate_b),
    )


def _filter_pairs(
    pairs: list[PairOut],
    *,
    exchanges: set[str] | None,
    symbols: set[str] | None,
    min_abs_corr: float,
    min_n_obs: int,
    only_directional: bool,
) -> list[PairOut]:
    out: list[PairOut] = []
    for p in pairs:
        if p.abs_correlation < min_abs_corr:
            continue
        if p.n_obs < min_n_obs:
            continue
        if only_directional and p.leader == "none":
            continue
        if exchanges is not None and (
            p.exchange_a not in exchanges and p.exchange_b not in exchanges
        ):
            continue
        if symbols is not None and p.symbol not in symbols:
            continue
        out.append(p)
    return out


_SORT_KEYS = {
    "edge_bps": lambda p: -p.edge_bps,
    "abs_corr": lambda p: -p.abs_correlation,
    "n_obs": lambda p: -float(p.n_obs),
    "lag_abs": lambda p: -abs(p.best_lag_seconds),
}


def _sort_pairs(pairs: list[PairOut], sort_by: str) -> list[PairOut]:
    key = _SORT_KEYS.get(sort_by) or _SORT_KEYS["edge_bps"]
    return sorted(pairs, key=key)


def _split_csv(value: str | None) -> set[str] | None:
    if value is None:
        return None
    items = {x.strip() for x in value.split(",") if x.strip()}
    return items or None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _build_lifespan(state: WebState) -> Any:
    """Build a FastAPI lifespan context manager for ``state``.

    The background refresh thread is started on entry and joined on exit so
    we never leak threads between hot-reload cycles in development.
    """

    stop_event = threading.Event()
    holder: dict[str, threading.Thread | None] = {"thread": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        thread = threading.Thread(
            target=_refresh_loop,
            args=(state, stop_event),
            name="lead-lag-scanner-web-refresh",
            daemon=True,
        )
        holder["thread"] = thread
        thread.start()
        try:
            yield
        finally:
            stop_event.set()
            if holder["thread"] is not None:
                holder["thread"].join(timeout=5.0)

    return lifespan


def _build_state_response(state: WebState) -> StateOut:
    with state.lock:
        snap = state.snapshot
    merged = merge_runtime_symbols(state.base_config)
    runtime = list(load_runtime_symbols(state.base_config.storage.data_dir))
    return StateOut(
        computed_at=snap.computed_at.isoformat(timespec="seconds"),
        refresh_seconds=state.refresh_seconds,
        last_compute_seconds=snap.last_compute_seconds,
        last_error=snap.last_error,
        n_trades=snap.n_trades,
        n_pairs=len(snap.results),
        exchanges_configured=list(merged.exchanges),
        exchanges_with_trades=list(snap.exchanges_with_trades),
        symbols_configured=list(merged.symbols),
        symbols_with_trades=list(snap.symbols_with_trades),
        runtime_symbols=runtime,
        taker_bps=merged.report.taker_bps,
    )


def _build_pairs_response(
    state: WebState,
    *,
    sort_by: str,
    exchanges: str | None,
    symbols: str | None,
    min_corr: float,
    min_obs: int,
    only_directional: bool,
    limit: int,
) -> PairsOut:
    with state.lock:
        snap = state.snapshot
    taker = state.base_config.report.taker_bps
    all_pairs = [_result_to_pair(r, taker) for r in snap.results]
    filtered = _filter_pairs(
        all_pairs,
        exchanges=_split_csv(exchanges),
        symbols=_split_csv(symbols),
        min_abs_corr=max(0.0, min(1.0, float(min_corr))),
        min_n_obs=max(0, int(min_obs)),
        only_directional=bool(only_directional),
    )
    ranked = _sort_pairs(filtered, sort_by)[: max(1, int(limit))]
    return PairsOut(
        computed_at=snap.computed_at.isoformat(timespec="seconds"),
        n_total=len(all_pairs),
        n_filtered=len(filtered),
        pairs=ranked,
        diagnostics=[
            DiagnosticOut(**asdict(d)) for d in sorted(snap.diagnostics, key=lambda x: x.exchange)
        ],
    )


def _add_runtime_symbol(state: WebState, payload: AddSymbolIn) -> SymbolsOut:
    canonical = _normalise_symbol(payload.symbol)
    existing = list(load_runtime_symbols(state.base_config.storage.data_dir))
    if canonical in state.base_config.symbols:
        return SymbolsOut(
            runtime_symbols=existing,
            message=f"{canonical} уже есть в основном списке.",
        )
    if canonical in existing:
        return SymbolsOut(
            runtime_symbols=existing,
            message=f"{canonical} уже добавлен.",
        )
    existing.append(canonical)
    save_runtime_symbols(state.base_config.storage.data_dir, tuple(existing))
    return SymbolsOut(
        runtime_symbols=existing,
        message=(
            f"{canonical} добавлен. Перезапустите `collect`, чтобы новые трейды начали поступать."
        ),
    )


def _remove_runtime_symbol(state: WebState, symbol: str) -> SymbolsOut:
    canonical = _normalise_symbol(symbol)
    existing = list(load_runtime_symbols(state.base_config.storage.data_dir))
    if canonical not in existing:
        raise HTTPException(status_code=404, detail=f"{canonical} не в runtime-списке.")
    new_list = [s for s in existing if s != canonical]
    save_runtime_symbols(state.base_config.storage.data_dir, tuple(new_list))
    return SymbolsOut(
        runtime_symbols=new_list,
        message=f"{canonical} удалён из runtime-списка.",
    )


def _force_refresh(state: WebState) -> dict[str, Any]:
    snap = _refresh_snapshot(state)
    with state.lock:
        state.snapshot = snap
    return {
        "ok": True,
        "computed_at": snap.computed_at.isoformat(timespec="seconds"),
        "n_trades": snap.n_trades,
        "n_pairs": len(snap.results),
        "last_error": snap.last_error,
    }


def create_app(config: Config, *, refresh_seconds: float | None = None) -> FastAPI:
    """Build a FastAPI app for the given :class:`Config`.

    A background thread refreshes the analyzer snapshot every
    ``refresh_seconds`` (defaulting to ``config.dashboard.refresh_seconds``).
    The app exposes ``/`` (the SPA), ``/api/state``, ``/api/pairs``, and
    ``/api/symbols`` (GET / POST / DELETE).
    """

    state = WebState(
        base_config=config,
        refresh_seconds=float(refresh_seconds or config.dashboard.refresh_seconds),
    )
    state.snapshot = _refresh_snapshot(state)

    app = FastAPI(
        title="lead-lag-scanner",
        version="0.1.0",
        description="Web dashboard for cross-exchange lead-lag analysis.",
        lifespan=_build_lifespan(state),
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> HTMLResponse:
        if not INDEX_HTML_PATH.exists():
            raise HTTPException(status_code=500, detail="frontend bundle is missing")
        return HTMLResponse(INDEX_HTML_PATH.read_text(encoding="utf-8"))

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        with state.lock:
            snap = state.snapshot
        return {
            "ok": True,
            "computed_at": snap.computed_at.isoformat(timespec="seconds"),
            "n_trades": snap.n_trades,
        }

    @app.get("/api/state", response_model=StateOut)
    def get_state() -> StateOut:
        return _build_state_response(state)

    @app.get("/api/pairs", response_model=PairsOut)
    def get_pairs(
        sort_by: str = "edge_bps",
        exchanges: str | None = None,
        symbols: str | None = None,
        min_corr: float = 0.0,
        min_obs: int = 0,
        only_directional: bool = False,
        limit: int = 200,
    ) -> PairsOut:
        return _build_pairs_response(
            state,
            sort_by=sort_by,
            exchanges=exchanges,
            symbols=symbols,
            min_corr=min_corr,
            min_obs=min_obs,
            only_directional=only_directional,
            limit=limit,
        )

    @app.get("/api/symbols", response_model=SymbolsOut)
    def get_symbols() -> SymbolsOut:
        runtime = list(load_runtime_symbols(state.base_config.storage.data_dir))
        return SymbolsOut(runtime_symbols=runtime)

    @app.post("/api/symbols", response_model=SymbolsOut)
    def add_symbol(payload: AddSymbolIn) -> SymbolsOut:
        return _add_runtime_symbol(state, payload)

    @app.delete("/api/symbols/{symbol:path}", response_model=SymbolsOut)
    def remove_symbol(symbol: str) -> SymbolsOut:
        return _remove_runtime_symbol(state, symbol)

    @app.post("/api/refresh")
    def force_refresh() -> JSONResponse:
        return JSONResponse(_force_refresh(state))

    return app


__all__ = [
    "WebState",
    "_Snapshot",
    "_filter_pairs",
    "_normalise_symbol",
    "_refresh_snapshot",
    "_sort_pairs",
    "create_app",
]
