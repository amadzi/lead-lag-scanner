"""Persistent storage for collected trades.

Each trade is normalised to::

    timestamp_ms       (int64)   — exchange matching-engine timestamp (ms)
    exchange           (str)
    symbol             (str)
    price              (float64)
    amount             (float64)
    side               (str)
    trade_id           (str)
    local_recv_ts_ns   (int64)   — collector wall-clock when the trade was
                                   received, in nanoseconds since epoch
                                   (0 if unknown, e.g. from legacy shards)

The ``local_recv_ts_ns`` column lets the analyzer (a) calibrate per-exchange
clock skew using ``median(local_recv − exchange_ts)``, (b) reason about
WebSocket / REST transport latency, and (c) drop trades whose
``|local − exchange_ts|`` is implausibly large.

Trades are buffered in memory and flushed to parquet files partitioned by
``exchange`` and ``date``::

    <data_dir>/trades/<exchange>/<YYYY-MM-DD>-<seq>.parquet

Each flush writes a brand-new immutable file. Files are first written to a
``.tmp`` sibling and atomically renamed into place via ``os.replace`` so a
concurrent reader (e.g. the live web dashboard) only ever sees a complete
parquet file with a valid footer. Legacy shards named
``<YYYY-MM-DD>.parquet`` (without a ``-<seq>`` suffix) are still picked up
by the read-side glob; new writes never touch them.

A DuckDB database is created on demand (``build_duckdb_view``) and exposes a
single view ``trades`` that unions every parquet shard. Old shards written
before ``local_recv_ts_ns`` was added are loaded with that column filled to
0; the analyzer treats 0 as "unknown" and silently skips that trade for
clock-skew calibration purposes.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

# Each parquet file we write is named ``YYYY-MM-DD-<8-digit-seq>.parquet``.
# The regex is anchored against the filename stem so we never accidentally
# parse a legacy ``YYYY-MM-DD.parquet`` shard as a sequence-suffixed one.
_SHARD_FILENAME_RE = re.compile(r"^(?P<day>\d{4}-\d{2}-\d{2})-(?P<seq>\d{8})$")

TRADE_SCHEMA = pa.schema(
    [
        pa.field("timestamp_ms", pa.int64()),
        pa.field("exchange", pa.string()),
        pa.field("symbol", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("amount", pa.float64()),
        pa.field("side", pa.string()),
        pa.field("trade_id", pa.string()),
        pa.field("local_recv_ts_ns", pa.int64()),
    ]
)

TRADE_COLUMNS: tuple[str, ...] = tuple(f.name for f in TRADE_SCHEMA)


@dataclass(slots=True)
class Trade:
    """Single trade event."""

    timestamp_ms: int
    exchange: str
    symbol: str
    price: float
    amount: float
    side: str
    trade_id: str
    local_recv_ts_ns: int = 0


@dataclass(slots=True)
class TradeWriter:
    """Buffered parquet writer that flushes immutable per-day shards.

    Each ``flush()`` writes a fresh ``<YYYY-MM-DD>-<8-digit-seq>.parquet``
    file via a temp-then-rename dance, so a concurrent reader (the live
    web dashboard) only ever observes complete parquet files with valid
    footers. Read paths simply glob ``trades/**/*.parquet``.

    Not thread-safe; create one writer per producer task.
    """

    data_dir: Path
    flush_every: int = 1000
    _buffer: list[Trade] = field(default_factory=list)
    # Per-(exchange, day) flush counter, kept in memory for the lifetime of
    # the writer. ``_resolve_seq_floor`` seeds it from any pre-existing
    # shards on disk so a restarted collector does not collide with files
    # written in a previous run.
    _seq: dict[tuple[str, str], int] = field(default_factory=dict)

    def append(self, trade: Trade) -> None:
        self._buffer.append(trade)
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        # Group by (exchange, date) so each shard is written as one new file.
        groups: dict[tuple[str, str], list[Trade]] = {}
        for tr in self._buffer:
            day = datetime.fromtimestamp(tr.timestamp_ms / 1000.0, tz=UTC).strftime("%Y-%m-%d")
            groups.setdefault((tr.exchange, day), []).append(tr)

        for (exchange, day), trades in groups.items():
            shard_dir = self.data_dir / "trades" / exchange
            shard_dir.mkdir(parents=True, exist_ok=True)
            seq = self._next_seq(shard_dir, exchange, day)
            final_path = shard_dir / f"{day}-{seq:08d}.parquet"
            tmp_path = shard_dir / f".{day}-{seq:08d}.parquet.tmp"
            table = _trades_to_table(trades)
            try:
                pq.write_table(table, tmp_path, compression="zstd")
                # ``os.replace`` is atomic on POSIX/NTFS for same-filesystem
                # paths, so a reader that opens ``final_path`` either sees
                # the previous file or the new one — never a half-written
                # blob with a missing footer.
                os.replace(tmp_path, final_path)
            except Exception:
                # If anything goes wrong (out-of-disk, perms), make sure we
                # don't leave a stale ``.tmp`` lying around to confuse later
                # debugging. We re-raise so the caller learns about it.
                if tmp_path.exists():
                    with contextlib.suppress(OSError):
                        tmp_path.unlink()
                raise
        self._buffer.clear()

    def close(self) -> None:
        self.flush()

    def _next_seq(self, shard_dir: Path, exchange: str, day: str) -> int:
        key = (exchange, day)
        if key not in self._seq:
            self._seq[key] = self._resolve_seq_floor(shard_dir, day)
        seq = self._seq[key]
        self._seq[key] = seq + 1
        return seq

    @staticmethod
    def _resolve_seq_floor(shard_dir: Path, day: str) -> int:
        """Return the smallest free sequence number for ``day`` in ``shard_dir``.

        Scans ``<day>-<seq>.parquet`` siblings and returns ``max(seq) + 1``,
        falling back to ``0`` when none exist. Legacy ``<day>.parquet``
        shards (without a sequence suffix) are intentionally ignored — the
        new writer never overwrites them, and they are still read by
        :func:`load_trades` via the directory glob.
        """

        max_seq = -1
        for path in shard_dir.glob(f"{day}-*.parquet"):
            match = _SHARD_FILENAME_RE.match(path.stem)
            if match is None:
                continue
            try:
                seq = int(match.group("seq"))
            except ValueError:
                continue
            max_seq = max(max_seq, seq)
        return max_seq + 1


def _read_shard_promoting(path: Path) -> pa.Table:
    """Read an existing shard and promote it to :data:`TRADE_SCHEMA`.

    Older shards (pre-``local_recv_ts_ns``) lack one or more of the columns
    in the current schema; we backfill missing columns with zero / empty
    string before concatenating against newly buffered rows.
    """

    existing = pq.read_table(path)
    if existing.schema.equals(TRADE_SCHEMA):
        return existing
    n = existing.num_rows
    columns: dict[str, pa.Array] = {name: existing.column(name) for name in existing.column_names}
    for f in TRADE_SCHEMA:
        if f.name in columns:
            continue
        if pa.types.is_integer(f.type):
            columns[f.name] = pa.array([0] * n, type=f.type)
        elif pa.types.is_floating(f.type):
            columns[f.name] = pa.array([0.0] * n, type=f.type)
        else:
            columns[f.name] = pa.array([""] * n, type=f.type)
    return pa.table(columns, schema=TRADE_SCHEMA)


def _trades_to_table(trades: list[Trade]) -> pa.Table:
    return pa.table(
        {
            "timestamp_ms": [t.timestamp_ms for t in trades],
            "exchange": [t.exchange for t in trades],
            "symbol": [t.symbol for t in trades],
            "price": [t.price for t in trades],
            "amount": [t.amount for t in trades],
            "side": [t.side for t in trades],
            "trade_id": [t.trade_id for t in trades],
            "local_recv_ts_ns": [t.local_recv_ts_ns for t in trades],
        },
        schema=TRADE_SCHEMA,
    )


def parquet_glob(data_dir: Path) -> str:
    """Return the glob string used by DuckDB to read every shard."""

    return str(data_dir / "trades" / "**" / "*.parquet")


def build_duckdb_view(data_dir: Path, duckdb_path: Path) -> None:
    """Create a DuckDB database with a ``trades`` view over the parquet shards."""

    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    glob = parquet_glob(data_dir)
    with duckdb.connect(str(duckdb_path)) as con:
        con.execute("DROP VIEW IF EXISTS trades")
        con.execute(
            f"CREATE VIEW trades AS SELECT * FROM read_parquet('{glob}', hive_partitioning=0)"
        )


# Match a parquet path embedded in a DuckDB error message. Two real-world
# variants we have to support:
#
#   "No magic bytes found at end of file 'data/trades/gate/2026-05-09.parquet'"
#   "Invalid Input Error: File '/abs/path/x.parquet' too small to be a Parquet file"
#
# Case-insensitive on the leading word so both forms match. We capture up to
# ``.parquet`` so paths containing other characters are still extracted
# correctly. Used by :func:`load_trades` to identify and quarantine the
# offending shard.
_CORRUPT_PATH_RE = re.compile(r"\bfile ['\"]([^'\"]+\.parquet)['\"]", re.IGNORECASE)


def _quarantine_shard(path: Path) -> bool:
    """Rename a corrupt parquet shard to ``<name>.parquet.corrupt``.

    Returns ``True`` on success, ``False`` if the source path no longer
    exists (e.g. a concurrent writer rotated it) or the rename fails. We
    keep the file on disk under a different extension so the user can
    inspect or recover it manually; the read-side glob (``*.parquet``)
    will simply skip it.
    """

    target = path.with_suffix(path.suffix + ".corrupt")
    try:
        os.replace(path, target)
    except FileNotFoundError:
        return False
    except OSError as exc:
        log.warning("could not quarantine corrupt shard %s: %s", path, exc)
        return False
    log.warning("quarantined corrupt shard %s -> %s", path, target)
    return True


def _quarantine_corrupt_shard_from_error(data_dir: Path, exc: BaseException) -> bool:
    """Parse a DuckDB error message and quarantine the offending parquet.

    Returns ``True`` when a shard was successfully renamed (so the caller
    knows it's worth retrying the query), ``False`` otherwise.
    """

    match = _CORRUPT_PATH_RE.search(str(exc))
    if match is None:
        return False
    raw = match.group(1)
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = (data_dir / raw).resolve() if (data_dir / raw).exists() else candidate
    if not candidate.exists():
        # Try the raw path as a relative path from the current working dir.
        cwd_candidate = Path.cwd() / raw
        if cwd_candidate.exists():
            candidate = cwd_candidate
        else:
            return False
    return _quarantine_shard(candidate)


def load_trades(
    data_dir: Path,
    *,
    exchange: str | None = None,
    symbol: str | None = None,
) -> pd.DataFrame:
    """Load trades from parquet into a pandas DataFrame.

    Returns an empty DataFrame with the correct columns if nothing has been
    collected yet. Old shards lacking the ``local_recv_ts_ns`` column are
    loaded with zeros via DuckDB's ``union_by_name=true``; missing columns
    are backfilled to 0 / "" so the returned frame always has the canonical
    schema.

    If a parquet shard has a corrupt footer (e.g. a legacy
    ``<YYYY-MM-DD>.parquet`` written in-place by a previous version of the
    code that was interrupted mid-write), the read fails with
    ``InvalidInputException``. We catch the error, parse the offending
    path out of the message, rename the file to ``<name>.parquet.corrupt``
    so the read-side glob skips it, and retry the query once. On a second
    failure we surface the original exception's text as part of the empty
    return path so callers (e.g. the web dashboard) can show an
    actionable message.
    """

    glob = parquet_glob(data_dir)
    query, params = _build_load_query(glob, exchange=exchange, symbol=symbol)
    # Up to ``len(corrupt_files) + 1`` attempts: each iteration may quarantine
    # a single bad shard and retry. We bound this with a small constant so a
    # pathological data dir can't trap us in a loop.
    for _attempt in range(8):
        try:
            with duckdb.connect() as con:
                return _post_process_frame(con.execute(query, params).df())
        except (duckdb.IOException, duckdb.InvalidInputException) as exc:
            if _quarantine_corrupt_shard_from_error(data_dir, exc):
                continue
            log.warning("load_trades failed; could not identify shard: %s", exc)
            return pd.DataFrame(columns=list(TRADE_COLUMNS))
    log.warning("load_trades: gave up after repeatedly quarantining shards")
    return pd.DataFrame(columns=list(TRADE_COLUMNS))


def _build_load_query(
    glob: str,
    *,
    exchange: str | None,
    symbol: str | None,
) -> tuple[str, list[str]]:
    query = f"SELECT * FROM read_parquet('{glob}', hive_partitioning=0, union_by_name=true)"
    params: list[str] = []
    filters: list[str] = []
    if exchange is not None:
        filters.append("exchange = ?")
        params.append(exchange)
    if symbol is not None:
        filters.append("symbol = ?")
        params.append(symbol)
    if filters:
        query += " WHERE " + " AND ".join(filters)
    return query, params


def _post_process_frame(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        return pd.DataFrame(columns=list(TRADE_COLUMNS))

    # Backfill missing canonical columns (older shards / partial schemas).
    for name in TRADE_COLUMNS:
        if name in df.columns:
            continue
        if name in {"local_recv_ts_ns", "timestamp_ms"}:
            df[name] = 0
        elif name in ("price", "amount"):
            df[name] = 0.0
        else:
            df[name] = ""
    # ``local_recv_ts_ns`` may be NULL on union'd reads when one shard had
    # the column and another didn't; coerce NULLs to 0 (= unknown).
    if bool(df["local_recv_ts_ns"].isna().any()):
        df["local_recv_ts_ns"] = df["local_recv_ts_ns"].fillna(0).astype("int64")
    return df
