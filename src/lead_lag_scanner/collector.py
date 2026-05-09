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
import random
import time
from dataclasses import dataclass
from typing import Any

import structlog

from .config import Config
from .exchanges import ExchangeHandle, close_handle, has_market, make_handle
from .storage import Trade, TradeWriter

log = structlog.get_logger(__name__)


_QUIET_HANDLER_INSTALLED = "_lead_lag_quiet_handler_installed"

# Substrings of asyncio default-handler messages we want to suppress entirely.
# These all describe transient ws/REST churn that the outer retry loop already
# handles; printing them as multi-line tracebacks per occurrence floods the
# terminal under high churn (a single 1006 close on 51 exchanges x 200 symbols
# can produce thousands of lines in a few seconds).
_SUPPRESSED_LOOP_MESSAGES: tuple[str, ...] = (
    "Future exception was never retrieved",
    "Task exception was never retrieved",
    "Exception in callback Client.receive_loop",
)

# Module prefixes whose unhandled errors are dropped silently. ccxt errors
# bubble up through asyncio when ccxt.pro spawns inner subscription tasks
# whose results we never directly await; the outer watch_trades retry loop in
# this file already logs and recovers from them, so re-printing is pure noise.
_SUPPRESSED_EXC_MODULES: tuple[str, ...] = ("ccxt.",)

# Path fragments that identify a frame as ccxt-internal. Some ccxt.pro
# clients raise built-in exceptions (e.g. ``AttributeError`` from
# ``ccxt/pro/htx.py:1921`` calling the non-existent ``client.reset(error)``)
# whose ``__module__`` is ``builtins`` — we still want to swallow them
# because they're library bugs that don't affect our pipeline.
_CCXT_FRAME_FRAGMENTS: tuple[str, ...] = (
    "/ccxt/",
    "\\ccxt\\",
    "/ccxt\\",
    "\\ccxt/",
)


def _exception_originates_in_ccxt(exc: BaseException) -> bool:
    """Return ``True`` when any frame in ``exc``'s traceback is from ccxt code."""

    tb = exc.__traceback__
    while tb is not None:
        filename = tb.tb_frame.f_code.co_filename
        if any(frag in filename for frag in _CCXT_FRAME_FRAGMENTS):
            return True
        tb = tb.tb_next
    return False


def install_quiet_loop_handler(loop: asyncio.AbstractEventLoop | None = None) -> None:
    """Replace the running loop's default exception handler with a quieter one.

    The default ``asyncio`` handler prints a full traceback for every
    Future/Task whose exception is never retrieved. ``ccxt.pro`` schedules
    short-lived inner tasks for ws subscriptions whose lifetime we do not
    directly own, so a routine ws disconnect (NetworkError 1006/1000), a
    schema mismatch on a single symbol (BadRequest from one exchange), or a
    library bug raising a built-in exception inside ccxt's own code (e.g.
    htx's ``client.reset(error)`` typo) ends up surfacing here even though
    our outer retry loop already handled it.

    The replacement handler:
      * silently drops messages matching :data:`_SUPPRESSED_LOOP_MESSAGES`
        when the underlying exception is either declared in ``ccxt.*`` or
        was raised inside a ``/ccxt/`` source file,
      * forwards everything else to the default handler so genuine bugs in
        our own code (TypeError, KeyError, etc.) still surface.

    Idempotent: calling it twice on the same loop is a no-op.
    """

    target = loop or asyncio.get_running_loop()
    if getattr(target, _QUIET_HANDLER_INSTALLED, False):
        return

    def _handler(loop_obj: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        msg = str(context.get("message", ""))
        exc = context.get("exception")
        if (
            exc is not None
            and any(token in msg for token in _SUPPRESSED_LOOP_MESSAGES)
            and (
                any(type(exc).__module__.startswith(p) for p in _SUPPRESSED_EXC_MODULES)
                or _exception_originates_in_ccxt(exc)
            )
        ):
            return
        loop_obj.default_exception_handler(context)

    target.set_exception_handler(_handler)
    # Mark via attribute so tests / repeated installs are idempotent.
    target.__dict__[_QUIET_HANDLER_INSTALLED] = True  # type: ignore[attr-defined]


@dataclass(slots=True)
class CollectionStats:
    trades_received: int = 0
    trades_written: int = 0
    errors: int = 0
    trades_dropped_drift: int = 0  # trades whose |local-exchange| > drift cap
    pairs_active: int = 0  # (exchange, symbol) actually being collected
    pairs_skipped: int = 0  # (exchange, symbol) skipped (not listed on exchange)


# Trades whose exchange-reported timestamp differs from our local receive
# clock by more than this many milliseconds are dropped at ingestion. Such
# trades are almost always API quirks (an exchange returning microseconds in
# a field documented as milliseconds, or a stale `since=` reset returning
# historical data) and they would otherwise poison clock-skew calibration.
# This is intentionally generous (1 hour) so that NTP-loose exchanges and
# WebSocket reconnect bursts are still kept.
_MAX_INGEST_DRIFT_MS: int = 3_600_000


# Substrings (case-insensitive, also matched against a quote-stripped copy
# of the message) that mark an error as *permanent* for a given (exchange,
# symbol) pair. When we see one we log the failure once, stop the loop, and
# never reschedule. Retrying these wastes CPU/network and floods the log
# with identical traceback noise.
#
# Examples:
#   - "requires apikey"         → e.g. luno's watch_trades demands auth.
#                                 Real ccxt string is `requires "apiKey"
#                                 credential`; the quote-stripped match
#                                 (see ``_is_permanent_error``) catches it
#                                 without us needing per-quote variants.
#   - "is not supported"        → exchange doesn't expose this method
#   - "notsupported"            → ccxt's NotSupported exception text
#   - "one symbol per instance" → cex.io ws limitation
#   - "protobuf"                → mexc switched ws frames to protobuf, ccxt
#                                 has not implemented the parser
#   - "403 forbidden" / "access denied" / "errors.edgesuite.net"
#                               → Akamai/CDN geo-block at our IP (e.g. bigone).
#                                 Will not recover by retrying.
#   - "invalid_argument" / "invalid contract"
#                               → e.g. weex returns this for symbols its ws
#                                 stream doesn't recognise — pure config
#                                 mismatch, no point retrying.
_PERMANENT_ERROR_PATTERNS: tuple[str, ...] = (
    "requires apikey",
    "is not supported",
    "notsupported",
    "one symbol per instance",
    "protobuf",
    "no such market",
    "market is not active",
    "subscribe to one symbol",
    "403 forbidden",
    "access denied",
    "errors.edgesuite.net",
    "invalid_argument",
    "invalid contract",
)


# Substrings (case-insensitive, also matched against a quote-stripped copy
# of the message) that mark an error as a *rate limit* hit. These are
# transient — the exchange will accept us again later — but the normal
# 1s/5s/15s/30s backoff is too aggressive: hammering on a 429 just extends
# the throttle window. We use a separate, longer schedule with jitter to
# spread out the retry storms when many concurrent symbol loops on the
# same exchange all hit 429 at once.
_RATE_LIMIT_PATTERNS: tuple[str, ...] = (
    "too many requests",
    "rate limit",
    '"code":429',
    "ratelimitexceeded",
    # bitget — uses a 30006 application-level code in addition to HTTP 429
    "request too many",
    '"code":30006',
    # bitmart — error codes for "subscribed message frequency" /
    # "subscribed total topic quantity" exceeding limits. Both clear after
    # a backoff window.
    '"errorcode":"90006"',
    '"errorcode":"90007"',
    "frequency exceeds limit",
    "topic quantity exceeds limit",
    # bigone application-level rate-limit code
    '"code":10429',
    # coinsph application-level rate-limit code
    '"code":-1003',
)


# Per-(exchange, symbol) "no progress" threshold. After this many
# *consecutive* errors with no successful trade batch in between we give up
# on the symbol forever. This handles the long-tail spam patterns that
# don't fit a single substring rule (upbit closing every ws on code 1000
# and 1006, kucoin's "Cannot write to closing transport", whitebit ws
# closures, mexc ping-pong timeouts) without needing exchange-specific
# heuristics. ``5`` chosen so a brief network blip (which would resolve in
# 1-2 retries) is forgiven, but a permanently-broken (exchange, symbol)
# pair is muted within ~80s for transient errors / ~13min for rate-limit
# errors. The counter resets to zero on every successful iteration.
_MAX_CONSECUTIVE_FAILURES: int = 5


# Exponential backoff schedule (seconds) for transient errors in the watch /
# fetch loops. We start at 1s, ramp up to 30s, and stay there. After any
# successful iteration the backoff resets to the start of the schedule so a
# brief blip doesn't penalise an exchange that recovered.
_TRANSIENT_BACKOFF_SECONDS: tuple[float, ...] = (1.0, 5.0, 15.0, 30.0)


# Backoff schedule (seconds, base) when the underlying error looks like a
# 429 / rate-limit. We start at 30s and stretch to 5 minutes; jitter is
# added on top so concurrent loops don't sync back into the next throttle
# window together. The values were chosen empirically to keep total log
# volume manageable when 28 symbols on the same exchange all hit 429
# simultaneously (typical cex/free tiers).
_RATE_LIMIT_BACKOFF_SECONDS: tuple[float, ...] = (30.0, 60.0, 120.0, 300.0)
_RATE_LIMIT_JITTER_FRACTION: float = 0.5


_QUOTE_CHARS = "\"'`"


def _normalise_error_text(exc: BaseException) -> tuple[str, str]:
    """Return ``(lower, lower_no_quotes)`` for substring matching.

    We match patterns against both the raw lowercased message and a copy
    with single/double/back quotes removed so that ccxt-style strings like
    ``requires "apiKey" credential`` match the canonical
    ``requires apikey`` pattern without needing per-quote variants.
    """

    lowered = str(exc).lower()
    no_quotes = lowered.translate(str.maketrans("", "", _QUOTE_CHARS))
    return lowered, no_quotes


def _is_permanent_error(exc: BaseException) -> bool:
    """Return ``True`` when ``exc`` matches a known unrecoverable pattern.

    The match is intentionally loose (case-insensitive substring, with
    quotes stripped) because different ccxt versions wrap the same
    underlying problem in slightly different error strings, and we'd
    rather skip a borderline case than spin on a hopeless retry forever.
    """

    lowered, no_quotes = _normalise_error_text(exc)
    return any(pattern in lowered or pattern in no_quotes for pattern in _PERMANENT_ERROR_PATTERNS)


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return ``True`` when ``exc`` looks like a transient 429 / rate-limit hit."""

    lowered, no_quotes = _normalise_error_text(exc)
    return any(pattern in lowered or pattern in no_quotes for pattern in _RATE_LIMIT_PATTERNS)


def _next_backoff(current_index: int, *, rate_limited: bool = False) -> tuple[float, int]:
    """Return ``(sleep_seconds, next_index)`` for the appropriate retry schedule.

    When ``rate_limited`` is ``True`` we draw from the longer 429-aware
    schedule and add up to ``±_RATE_LIMIT_JITTER_FRACTION`` of multiplicative
    jitter so concurrent loops don't all retry at the same instant.
    """

    schedule = _RATE_LIMIT_BACKOFF_SECONDS if rate_limited else _TRANSIENT_BACKOFF_SECONDS
    idx = min(current_index, len(schedule) - 1)
    base = schedule[idx]
    if rate_limited:
        # Multiplicative jitter in [1 - f, 1 + f]. ``random.random()`` is
        # fine here — these sleeps are not security-sensitive.
        jitter = 1.0 + (random.random() * 2.0 - 1.0) * _RATE_LIMIT_JITTER_FRACTION
        base *= jitter
    return base, idx + 1


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
    backoff_idx = 0
    consecutive_failures = 0
    while not stop_event.is_set():
        try:
            trades: list[dict[str, Any]] = await client.watch_trades(symbol)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            stats.errors += 1
            consecutive_failures += 1
            if _is_permanent_error(exc):
                # Log once at INFO and stop forever — there is no value in
                # retrying a NotSupported / apiKey-required / protobuf-frame
                # error every second on this (exchange, symbol) pair.
                log.info(
                    "watch_trades permanently disabled for symbol",
                    exchange=handle.exchange_id,
                    symbol=symbol,
                    error=f"{type(exc).__name__}: {exc}",
                )
                return
            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                # Give up on this (exchange, symbol) pair: we've burned the
                # full backoff schedule without ever seeing a successful
                # batch, so further retries would just add log noise.
                # Catches upbit's 1000-on-keepalive cycle, kucoin's
                # "Cannot write to closing transport", whitebit ws closures,
                # and mexc ping-pong timeouts without per-exchange knowledge.
                log.info(
                    "watch_trades disabled after consecutive failures",
                    exchange=handle.exchange_id,
                    symbol=symbol,
                    consecutive=consecutive_failures,
                    last_error=str(exc),
                )
                return
            rate_limited = _is_rate_limit_error(exc)
            sleep_for, backoff_idx = _next_backoff(backoff_idx, rate_limited=rate_limited)
            log.warning(
                "watch_trades error",
                exchange=handle.exchange_id,
                symbol=symbol,
                error=str(exc),
                retry_in_s=round(sleep_for, 1),
                rate_limited=rate_limited,
                consecutive=consecutive_failures,
            )
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=sleep_for)
                return
            except TimeoutError:
                continue
        # ccxt.pro delivers trades in batches; capture the local clock once
        # per batch so that trades sharing a single network frame share their
        # ``local_recv_ts_ns`` and clock-skew calibration is not biased by
        # how the analyzer iterates the batch.
        local_recv_ns = time.time_ns()
        # Successful iteration: reset the transient backoff schedule and the
        # consecutive-failure counter so a one-off blip earlier in the run
        # doesn't keep us in slow mode (or one error away from being muted).
        backoff_idx = 0
        consecutive_failures = 0
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
    backoff_idx = 0
    consecutive_failures = 0
    while not stop_event.is_set():
        try:
            trades: list[dict[str, Any]] = await client.fetch_trades(symbol, since=last_ts)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            stats.errors += 1
            consecutive_failures += 1
            if _is_permanent_error(exc):
                log.info(
                    "fetch_trades permanently disabled for symbol",
                    exchange=handle.exchange_id,
                    symbol=symbol,
                    error=f"{type(exc).__name__}: {exc}",
                )
                return
            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                log.info(
                    "fetch_trades disabled after consecutive failures",
                    exchange=handle.exchange_id,
                    symbol=symbol,
                    consecutive=consecutive_failures,
                    last_error=str(exc),
                )
                return
            rate_limited = _is_rate_limit_error(exc)
            sleep_for, backoff_idx = _next_backoff(backoff_idx, rate_limited=rate_limited)
            # REST loops already waited ``poll_interval`` between successful
            # iterations, so on non-rate-limit errors we overlay the transient
            # backoff on top of that base interval. For rate-limit errors we
            # use the schedule as-is — it's already much longer than 2x the
            # poll interval and we don't want to inflate it further.
            if not rate_limited:
                sleep_for = max(sleep_for, poll_interval * 2)
            log.warning(
                "fetch_trades error",
                exchange=handle.exchange_id,
                symbol=symbol,
                error=str(exc),
                retry_in_s=round(sleep_for, 1),
                rate_limited=rate_limited,
                consecutive=consecutive_failures,
            )
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=sleep_for)
                return
            except TimeoutError:
                continue
        local_recv_ns = time.time_ns()
        backoff_idx = 0
        consecutive_failures = 0
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
        # Demoted to debug: with 51 exchanges x 200 symbols this used to spam
        # ~7000 "skipping" lines at startup. The aggregate pairs_active /
        # pairs_skipped counters are surfaced once at startup instead.
        stats.pairs_skipped += 1
        log.debug(
            "skipping symbol - not listed on exchange",
            exchange=handle.exchange_id,
            symbol=symbol,
        )
        return
    stats.pairs_active += 1
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
    install_quiet_loop_handler()
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
