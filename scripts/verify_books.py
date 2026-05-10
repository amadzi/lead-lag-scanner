"""Verify lead-lag candidates against L1 orderbook (best bid / best ask).

The main analyzer measures lead-lag on the **trade-print stream**: it
correlates 1-second resampled last-trade prices across exchanges. Trade
prints can arrive batched or delayed by the exchange API even when the
real market state (best bid / best ask) updated immediately. So a
"binance leads bingx by 2 seconds" result from the analyzer might be
the *real* executable lag, or it might be a pure trade-stream batching
artifact that disappears the moment you look at quote updates.

This script answers that question by:

1. Subscribing to L1 orderbook (best bid + best ask) updates over
   WebSockets via ``ccxt.pro`` for each unique ``(exchange, symbol)``
   that appears in the candidate list.
2. Stamping every quote update with a local nanosecond receive timestamp.
3. After a configurable duration, resampling the mid-price
   (``(bid+ask)/2``) onto a fine grid (default 100 ms) and computing
   the cross-correlation between leader and follower over a tight lag
   grid (default ±2.5 s with 50 ms step).
4. Printing a verdict per candidate:

   - ``CONFIRMED``        — book lag's sign matches the trade-stream
     expectation *and* magnitude is within 1 second of expectation.
   - ``SIGN_OK``          — book lag sign matches expectation but
     magnitude differs significantly.
   - ``BOOK_NO_LAG``      — book lag is essentially zero (|lag| <
     200 ms): the trade-stream lag is almost certainly a batching
     artifact, not executable alpha.
   - ``FLIPPED``          — book lag has the opposite sign of the
     trade-stream expectation: the analyzer's leader/follower were
     misidentified (or the signal is pure noise).
   - ``no data`` / ``low N`` — not enough quote updates to decide.

Run it standalone, no need to touch the main collector:

.. code-block:: bash

    cd ~/lead-lag-scanner
    uv pip install "ccxt[pro]>=4.4.0"  # if not already installed via [ws] extra
    uv run python scripts/verify_books.py --duration 1800 --output verify-books.csv

While it runs, the existing ``collect`` and ``web`` commands are
unaffected; this script writes to its own CSV and reads no
``data/trades/`` shards.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    import ccxt.pro as ccxtpro
except ImportError:
    print(
        "Missing dependency: install with `uv pip install 'ccxt[pro]>=4.4.0'`",
        file=sys.stderr,
    )
    sys.exit(2)


# ---------------------------------------------------------------------------
# Candidate list
# ---------------------------------------------------------------------------
#
# Picked from the 6-hour run shared by the user. Selection criteria:
#
# - ``n_obs >= 200`` (enough samples for the trade-stream lag to be stable)
# - CI width <= 2 s (lag stable across resamples)
# - peak/zero-lag correlation ratio >= 1.6 (signal not dominated by market
#   beta — the correlation at lag = 0 is meaningfully smaller than at peak)
# - ``|best_lag| < 4`` s (not at the analyzer's ±5 s grid edge)
# - ``leader`` is a venue that *plausibly* leads (binance / okx / bybit
#   for BTC/ETH/majors; for alts a smaller venue can lead if its quote
#   discovery is faster — those are flagged "investigate")
#
# Pairs where a low-volume venue (toobit, bitmart, deepcoin) "leads" a
# tier-1 venue on BTC/ETH/SOL are NOT included — those are almost
# certainly websocket-batching artifacts and should be rejected without
# spending the verification budget on them.


@dataclass(frozen=True)
class Candidate:
    """One ``(symbol, leader, follower)`` triplet to verify."""

    symbol: str
    leader: str
    follower: str
    expected_lag_s: float  # signed; positive means follower lags leader


# Top picks (strongest evidence in the trade-stream analyzer).
CANDIDATES: tuple[Candidate, ...] = (
    # 1. Strongest single signal in the 6h run: corr 0.943, N 1994,
    #    peak/zero-lag corr ratio ~3x. If this doesn't show up in book
    #    data, nothing will.
    Candidate("SUI/USDT", "binance", "bingx", +2.0),
    # 2-7. okx → gate cluster. Same lag (-1 s) across 6 different
    #      symbols with tight CIs. Strongly suggests gate's trade-stream
    #      publishes ~1 s later than okx's. Need to verify whether
    #      gate's *quote* stream is also ~1 s late — that's the real test.
    Candidate("SUI/USDT", "okx", "gate", -1.0),
    Candidate("DOGE/USDT", "okx", "gate", -1.0),
    Candidate("UNI/USDT", "okx", "gate", -1.0),
    Candidate("ONDO/USDT", "okx", "gate", -1.0),
    Candidate("SOL/USDT", "okx", "gate", -1.0),
    Candidate("TON/USDT", "okx", "gate", -1.0),
    # 8. okx → bitmart for SUI (same pattern at -1 s, N 2240)
    Candidate("SUI/USDT", "okx", "bitmart", -1.0),
    # 9-10. Other directional patterns worth checking
    Candidate("LDO/USDT", "woo", "bitmart", -1.0),
    Candidate("ETH/USDT", "gate", "woo", +1.0),
    # 11. Small N but extreme peak/zero ratio (5.6x): is the signal real?
    Candidate("NOT/USDT", "deepcoin", "gate", +1.0),
)


# Some ccxt.pro implementations don't expose `watch_order_book`. The
# polling fallback uses `fetch_order_book` over REST at a fixed cadence
# — slower than ws, but better than nothing.
REST_POLL_INTERVAL_S = 0.5


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def _extract_top_of_book(ob: dict[str, Any]) -> tuple[float, float] | None:
    """Return ``(bid, ask)`` from a ccxt order-book dict, or None if degenerate."""

    bids = ob.get("bids") or []
    asks = ob.get("asks") or []
    if not bids or not asks:
        return None
    bid = float(bids[0][0])
    ask = float(asks[0][0])
    if bid <= 0 or ask <= 0 or ask < bid:
        return None
    return bid, ask


async def _watch_book_loop(
    exchange_id: str,
    symbol: str,
    queue: asyncio.Queue[tuple[Any, ...]],
    stop_event: asyncio.Event,
) -> None:
    """Subscribe to one ``(exchange, symbol)`` L1 book stream.

    Falls back to REST polling when ``watch_order_book`` is unavailable
    or fails repeatedly. Reports errors to stderr at a throttled rate
    so a single misbehaving venue can't flood the terminal.
    """

    if not hasattr(ccxtpro, exchange_id):
        print(
            f"[{exchange_id}] no ccxt.pro client; skipping {symbol}",
            file=sys.stderr,
        )
        return

    ex = getattr(ccxtpro, exchange_id)({"enableRateLimit": True})
    pro_supported = True
    consecutive_errors = 0
    try:
        try:
            await ex.load_markets()
        except Exception as exc:
            print(
                f"[{exchange_id}] load_markets failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return
        if symbol not in ex.markets:
            print(f"[{exchange_id}] {symbol} not listed; skipping", file=sys.stderr)
            return

        while not stop_event.is_set():
            try:
                if pro_supported and getattr(ex, "has", {}).get("watchOrderBook", False):
                    ob = await ex.watch_order_book(symbol, limit=5)
                else:
                    ob = await ex.fetch_order_book(symbol, limit=5)
                    await asyncio.sleep(REST_POLL_INTERVAL_S)
                top = _extract_top_of_book(ob)
                if top is None:
                    continue
                bid, ask = top
                ts_ns = time.time_ns()
                mid = 0.5 * (bid + ask)
                exch_ts_ms = ob.get("timestamp")
                await queue.put((ts_ns, exch_ts_ms, exchange_id, symbol, bid, ask, mid))
                consecutive_errors = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                consecutive_errors += 1
                if consecutive_errors == 5 and pro_supported:
                    print(
                        f"[{exchange_id} {symbol}] ws errors >=5; switching to REST polling",
                        file=sys.stderr,
                    )
                    pro_supported = False
                if consecutive_errors <= 3 or consecutive_errors % 50 == 0:
                    print(
                        f"[{exchange_id} {symbol}] error #{consecutive_errors}: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                await asyncio.sleep(min(consecutive_errors, 10))
    finally:
        with contextlib.suppress(Exception):
            await ex.close()


async def _writer_loop(
    queue: asyncio.Queue[tuple[Any, ...]],
    output_path: str,
    stop_event: asyncio.Event,
) -> int:
    """Drain ``queue`` to ``output_path`` (CSV). Returns rows written."""

    rows_written = 0
    last_progress_at = time.time()
    with open(output_path, "w", newline="", buffering=1) as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["ts_ns_local", "ts_ms_exchange", "exchange", "symbol", "bid", "ask", "mid"]
        )
        while not (stop_event.is_set() and queue.empty()):
            try:
                row = await asyncio.wait_for(queue.get(), timeout=1.0)
            except TimeoutError:
                continue
            writer.writerow(row)
            rows_written += 1
            now = time.time()
            if now - last_progress_at > 30:
                last_progress_at = now
                fh.flush()
                print(f"[writer] {rows_written} rows written so far")
    return rows_written


async def _collect(duration: float, output_path: str) -> None:
    """Spawn watchers + writer, wait ``duration`` seconds, shut down cleanly."""

    queue: asyncio.Queue[tuple[Any, ...]] = asyncio.Queue(maxsize=50_000)
    stop_event = asyncio.Event()

    # Build the unique (exchange, symbol) set across all candidates.
    targets: set[tuple[str, str]] = set()
    for c in CANDIDATES:
        targets.add((c.leader, c.symbol))
        targets.add((c.follower, c.symbol))

    print(
        f"Subscribing to {len(targets)} L1 orderbook streams "
        f"({len(CANDIDATES)} candidate pairs) for {duration:.0f}s..."
    )

    write_task = asyncio.create_task(_writer_loop(queue, output_path, stop_event))
    watch_tasks = [
        asyncio.create_task(_watch_book_loop(ex, sym, queue, stop_event))
        for ex, sym in sorted(targets)
    ]

    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop_event.set)

    try:
        await asyncio.wait_for(stop_event.wait(), timeout=duration)
    except TimeoutError:
        pass
    finally:
        stop_event.set()
        for t in watch_tasks:
            t.cancel()
        with contextlib.suppress(BaseException):
            await asyncio.gather(*watch_tasks, return_exceptions=True)
        # let writer drain remaining queue items
        with contextlib.suppress(BaseException):
            rows = await asyncio.wait_for(write_task, timeout=10)
            print(f"[writer] done: {rows} rows total")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _cross_corr_grid(
    leader: pd.Series,
    follower: pd.Series,
    *,
    resample_ms: int,
    max_lag_s: float,
    lag_step_s: float,
) -> tuple[float, float]:
    """Return ``(best_lag_s, best_signed_corr)`` over the lag grid.

    Positive ``best_lag_s`` means ``follower`` reacts to ``leader``
    AFTER ``leader`` moves. Negative means the other way around.
    """

    aligned_a, aligned_b = leader.align(follower, join="inner")
    if len(aligned_a) < 100:
        return float("nan"), float("nan")

    # Drop the index *before* slicing: ``pd.Series.corr`` aligns by index,
    # so ``a.iloc[:-shift].corr(b.iloc[shift:])`` would silently re-align
    # the two slices and degenerate to ``a.corr(b)`` because the surviving
    # timestamp intersection is essentially the same after the shift.
    # ``np.corrcoef`` on raw numpy arrays does the positional comparison
    # we actually want.
    arr_a = aligned_a.to_numpy()
    arr_b = aligned_b.to_numpy()

    lags = np.arange(-max_lag_s, max_lag_s + lag_step_s, lag_step_s)
    best_corr = 0.0
    best_lag = 0.0
    for lag_s in lags:
        shift = round(float(lag_s) * 1000 / resample_ms)
        if shift > 0:
            x, y = arr_a[:-shift], arr_b[shift:]
        elif shift < 0:
            x, y = arr_a[-shift:], arr_b[:shift]
        else:
            x, y = arr_a, arr_b
        if len(x) < 50 or np.std(x) == 0 or np.std(y) == 0:
            continue
        corr = float(np.corrcoef(x, y)[0, 1])
        if np.isnan(corr):
            continue
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = float(lag_s)
    return best_lag, best_corr


def _verdict(
    *,
    expected_lag_s: float,
    book_lag_s: float,
    book_corr: float,
    zero_lag_tolerance_s: float = 0.2,
    mag_tolerance_s: float = 1.0,
    min_corr: float = 0.2,
) -> str:
    """Label the comparison between trade-lag expectation and book-lag result."""

    if abs(book_corr) < min_corr:
        return "low_corr"
    if abs(book_lag_s) < zero_lag_tolerance_s:
        return "BOOK_NO_LAG (fake)"
    sign_match = (book_lag_s > 0) == (expected_lag_s > 0)
    if not sign_match:
        return "FLIPPED (fake)"
    if abs(abs(book_lag_s) - abs(expected_lag_s)) < mag_tolerance_s:
        return "CONFIRMED"
    return "SIGN_OK"


def _build_mid_grids(
    csv_path: str,
    *,
    resample_ms: int,
) -> dict[tuple[str, str], pd.Series]:
    """Load CSV → resampled forward-filled mid-return series per (ex, symbol)."""

    df = pd.read_csv(csv_path)
    if df.empty:
        return {}
    df["ts"] = pd.to_datetime(df["ts_ns_local"], unit="ns", utc=True)
    df = df.set_index("ts").sort_index()

    grids: dict[tuple[str, str], pd.Series] = {}
    for (ex, sym), group in df.groupby(["exchange", "symbol"]):
        mid = group["mid"].resample(f"{resample_ms}ms").last().ffill()
        ret = mid.pct_change().dropna()
        if len(ret) >= 100:
            grids[(str(ex), str(sym))] = ret
    return grids


def analyze(
    csv_path: str,
    *,
    resample_ms: int = 100,
    max_lag_s: float = 2.5,
    lag_step_s: float = 0.05,
) -> None:
    """Print a verdict table for each candidate."""

    if not os.path.exists(csv_path):
        print(f"No CSV at {csv_path}; nothing to analyze.")
        return
    grids = _build_mid_grids(csv_path, resample_ms=resample_ms)
    if not grids:
        print("No usable data (need >= 100 samples per stream).")
        return

    header = (
        f"{'symbol':<12} {'leader':<10} {'follower':<10} "
        f"{'exp_s':>6} {'book_s':>7} {'corr':>7} {'n':>6}  verdict"
    )
    print()
    print(header)
    print("-" * len(header))
    for c in CANDIDATES:
        a = grids.get((c.leader, c.symbol))
        b = grids.get((c.follower, c.symbol))
        if a is None or b is None:
            n_a = len(a) if a is not None else 0
            n_b = len(b) if b is not None else 0
            print(
                f"{c.symbol:<12} {c.leader:<10} {c.follower:<10} "
                f"{c.expected_lag_s:>+6.1f} {'-':>7} {'-':>7} "
                f"{f'{n_a}/{n_b}':>6}  no_data"
            )
            continue
        best_lag, best_corr = _cross_corr_grid(
            a,
            b,
            resample_ms=resample_ms,
            max_lag_s=max_lag_s,
            lag_step_s=lag_step_s,
        )
        n = int(min(len(a), len(b)))
        if pd.isna(best_lag) or pd.isna(best_corr):
            print(
                f"{c.symbol:<12} {c.leader:<10} {c.follower:<10} "
                f"{c.expected_lag_s:>+6.1f} {'-':>7} {'-':>7} {n:>6}  low_n"
            )
            continue
        verdict = _verdict(
            expected_lag_s=c.expected_lag_s,
            book_lag_s=best_lag,
            book_corr=best_corr,
        )
        print(
            f"{c.symbol:<12} {c.leader:<10} {c.follower:<10} "
            f"{c.expected_lag_s:>+6.1f} {best_lag:>+7.2f} {best_corr:>+7.3f} "
            f"{n:>6}  {verdict}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Verify trade-stream lead-lag candidates against L1 orderbook "
            "(best bid / best ask) via WebSockets."
        ),
    )
    p.add_argument(
        "--duration",
        type=float,
        default=1800.0,
        help="seconds to record book updates (default: 1800 = 30 min)",
    )
    p.add_argument(
        "--output",
        type=str,
        default="verify-books.csv",
        help="path for the CSV of L1 quote snapshots (default: verify-books.csv)",
    )
    p.add_argument(
        "--analyze-only",
        action="store_true",
        help="skip collection; only analyze an existing CSV at --output",
    )
    p.add_argument(
        "--resample-ms",
        type=int,
        default=100,
        help="resample mid-price to this grid (ms) before correlating (default: 100)",
    )
    p.add_argument(
        "--max-lag-s",
        type=float,
        default=2.5,
        help="search lag grid spans ±max-lag-s (default: 2.5)",
    )
    p.add_argument(
        "--lag-step-s",
        type=float,
        default=0.05,
        help="lag grid step in seconds (default: 0.05)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.analyze_only:
        asyncio.run(_collect(args.duration, args.output))
        print("\nCollection complete. Running analysis...\n")
    analyze(
        args.output,
        resample_ms=args.resample_ms,
        max_lag_s=args.max_lag_s,
        lag_step_s=args.lag_step_s,
    )


if __name__ == "__main__":
    main()
