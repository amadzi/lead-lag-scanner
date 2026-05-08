"""Async public-trade collector.

For each (exchange, symbol) the collector spawns one task that either:

* uses ``ccxt.pro``'s ``watchTrades`` when available, or
* falls back to polling ``fetchTrades`` over REST.

Trades are forwarded to a single :class:`TradeWriter` which buffers and
persists them to parquet. The collector exits cleanly after
``CollectorConfig.duration`` seconds or when cancelled.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from typing import Any

import structlog

from .config import Config
from .exchanges import ExchangeHandle, close_handle, has_market, make_handle
from .storage import Trade, TradeWriter

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class CollectionStats:
    trades_received: int = 0
    trades_written: int = 0
    errors: int = 0
    trades_dropped_drift: int = 0  # trades whose |local-exchange| > drift cap


# Trades whose exchange-reported timestamp differs from our local receive
# clock by more than this many milliseconds are dropped at ingestion. Such
# trades are almost always API quirks (an exchange returning microseconds in
# a field documented as milliseconds, or a stale `since=` reset returning
# historical data) and they would otherwise poison clock-skew calibration.
# This is intentionally generous (1 hour) so that NTP-loose exchanges and
# WebSocket reconnect bursts are still kept.
_MAX_INGEST_DRIFT_MS: int = 3_600_000


def _normalise_trade(
    exchange_id: str,
    symbol: str,
    raw: dict[str, Any],
    local_recv_ts_ns: int,
) -> Trade | None:
    """Convert a ccxt trade dict into our internal :class:`Trade` shape.

    Returns ``None`` when the trade is missing fields we cannot recover from
    or has a timestamp that drifts implausibly far from ``local_recv_ts_ns``.
    """

    ts = raw.get("timestamp")
    price = raw.get("price")
    amount = raw.get("amount")
    if ts is None or price is None or amount is None:
        return None
    side = raw.get("side") or "unknown"
    trade_id = raw.get("id") or ""
    try:
        ts_ms = int(ts)
        local_recv_ms = local_recv_ts_ns // 1_000_000
        if abs(local_recv_ms - ts_ms) > _MAX_INGEST_DRIFT_MS:
            return None
        return Trade(
            timestamp_ms=ts_ms,
            exchange=exchange_id,
            symbol=symbol,
            price=float(price),
            amount=float(amount),
            side=str(side),
            trade_id=str(trade_id),
            local_recv_ts_ns=local_recv_ts_ns,
        )
    except (TypeError, ValueError):
        return None


async def _watch_trades_loop(
    handle: ExchangeHandle,
    symbol: str,
    writer: TradeWriter,
    stats: CollectionStats,
    stop_event: asyncio.Event,
) -> None:
    client = handle.client
    seen: set[str] = set()
    while not stop_event.is_set():
        try:
            trades: list[dict[str, Any]] = await client.watch_trades(symbol)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            stats.errors += 1
            log.warning(
                "watch_trades error",
                exchange=handle.exchange_id,
                symbol=symbol,
                error=str(exc),
            )
            await asyncio.sleep(1.0)
            continue
        # ccxt.pro delivers trades in batches; capture the local clock once
        # per batch so that trades sharing a single network frame share their
        # ``local_recv_ts_ns`` and clock-skew calibration is not biased by
        # how the analyzer iterates the batch.
        local_recv_ns = time.time_ns()
        for raw in trades:
            tr = _normalise_trade(handle.exchange_id, symbol, raw, local_recv_ns)
            if tr is None:
                if raw.get("timestamp") is not None:
                    stats.trades_dropped_drift += 1
                continue
            key = f"{tr.timestamp_ms}-{tr.trade_id}-{tr.price}-{tr.amount}"
            if key in seen:
                continue
            seen.add(key)
            stats.trades_received += 1
            writer.append(tr)
            stats.trades_written += 1
        if len(seen) > 100_000:
            seen = set(list(seen)[-50_000:])


async def _fetch_trades_loop(
    handle: ExchangeHandle,
    symbol: str,
    writer: TradeWriter,
    stats: CollectionStats,
    stop_event: asyncio.Event,
    poll_interval: float,
) -> None:
    client = handle.client
    seen: set[str] = set()
    # Seed `since` to ~5 seconds ago so we don't pull historical archives from
    # exchanges (e.g. Kraken) that default to the first-ever trade.
    last_ts: int = int(time.time() * 1000) - 5_000
    while not stop_event.is_set():
        try:
            trades: list[dict[str, Any]] = await client.fetch_trades(symbol, since=last_ts)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            stats.errors += 1
            log.warning(
                "fetch_trades error",
                exchange=handle.exchange_id,
                symbol=symbol,
                error=str(exc),
            )
            await asyncio.sleep(poll_interval * 2)
            continue
        local_recv_ns = time.time_ns()
        for raw in trades:
            tr = _normalise_trade(handle.exchange_id, symbol, raw, local_recv_ns)
            if tr is None:
                if raw.get("timestamp") is not None:
                    stats.trades_dropped_drift += 1
                continue
            key = f"{tr.timestamp_ms}-{tr.trade_id}-{tr.price}-{tr.amount}"
            if key in seen:
                continue
            seen.add(key)
            last_ts = max(last_ts, tr.timestamp_ms)
            stats.trades_received += 1
            writer.append(tr)
            stats.trades_written += 1
        # Bound the dedup set so memory stays flat over long runs.
        if len(seen) > 100_000:
            seen = set(list(seen)[-50_000:])
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)


async def _run_for_symbol(
    handle: ExchangeHandle,
    symbol: str,
    writer: TradeWriter,
    stats: CollectionStats,
    stop_event: asyncio.Event,
    rest_poll_interval: float,
) -> None:
    if not await has_market(handle, symbol):
        log.info(
            "skipping symbol - not listed on exchange",
            exchange=handle.exchange_id,
            symbol=symbol,
        )
        return
    if handle.supports_watch_trades:
        await _watch_trades_loop(handle, symbol, writer, stats, stop_event)
    else:
        await _fetch_trades_loop(handle, symbol, writer, stats, stop_event, rest_poll_interval)


def _instantiate_handles(config: Config) -> list[ExchangeHandle]:
    handles: list[ExchangeHandle] = []
    for exchange_id in config.exchanges:
        try:
            handle = make_handle(
                exchange_id,
                prefer_websocket=config.collector.prefer_websocket,
                rate_limit_safety=config.collector.rate_limit_safety,
            )
        except ValueError as exc:
            log.warning("skipping unsupported exchange", exchange=exchange_id, error=str(exc))
            continue
        except Exception as exc:
            # ccxt constructors can raise a variety of errors (auth-required,
            # network, version mismatch). We never want one bad exchange to
            # abort the whole run.
            log.warning(
                "exchange constructor failed; skipping",
                exchange=exchange_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            continue
        handles.append(handle)
    return handles


def _spawn_tasks(
    config: Config,
    handles: list[ExchangeHandle],
    writer: TradeWriter,
    stats: CollectionStats,
    stop_event: asyncio.Event,
) -> list[asyncio.Task[None]]:
    tasks: list[asyncio.Task[None]] = []
    for handle in handles:
        for symbol in config.symbols:
            task = asyncio.create_task(
                _run_for_symbol(
                    handle,
                    symbol,
                    writer,
                    stats,
                    stop_event,
                    config.collector.rest_poll_interval,
                ),
                name=f"collect-{handle.exchange_id}-{symbol}",
            )
            tasks.append(task)
    return tasks


async def _await_deadline(stop_event: asyncio.Event, deadline: float) -> None:
    while time.monotonic() < deadline:
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=1.0)
            return
        except TimeoutError:
            continue


async def _shutdown(
    tasks: list[asyncio.Task[None]],
    handles: list[ExchangeHandle],
    writer: TradeWriter,
    stop_event: asyncio.Event,
) -> None:
    stop_event.set()
    for task in tasks:
        task.cancel()
    for task in tasks:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
    for handle in handles:
        try:
            await close_handle(handle)
        except Exception as exc:
            log.warning("error closing handle", exchange=handle.exchange_id, error=str(exc))
    writer.close()


async def collect(config: Config) -> CollectionStats:
    """Run the collector for ``config.collector.duration`` seconds.

    Returns aggregate :class:`CollectionStats` once finished.
    """

    config.storage.data_dir.mkdir(parents=True, exist_ok=True)
    writer = TradeWriter(data_dir=config.storage.data_dir)
    stats = CollectionStats()
    stop_event = asyncio.Event()

    handles = _instantiate_handles(config)
    if not handles:
        log.error("no usable exchanges; aborting collection")
        return stats

    tasks = _spawn_tasks(config, handles, writer, stats, stop_event)
    deadline = time.monotonic() + config.collector.duration
    try:
        await _await_deadline(stop_event, deadline)
    finally:
        await _shutdown(tasks, handles, writer, stop_event)

    return stats
