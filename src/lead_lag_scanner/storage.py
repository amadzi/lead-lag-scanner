"""Persistent storage for collected trades.

Each trade is normalised to::

    timestamp_ms (int64)   exchange (str)   symbol (str)   price (float64)
    amount (float64)       side (str)       trade_id (str)

Trades are buffered in memory and flushed to parquet files partitioned by
``exchange`` and ``date``::

    <data_dir>/trades/<exchange>/<YYYY-MM-DD>.parquet

A DuckDB database is created on demand (`build_duckdb_view`) and exposes a
single view ``trades`` that unions every parquet shard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TRADE_SCHEMA = pa.schema(
    [
        pa.field("timestamp_ms", pa.int64()),
        pa.field("exchange", pa.string()),
        pa.field("symbol", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("amount", pa.float64()),
        pa.field("side", pa.string()),
        pa.field("trade_id", pa.string()),
    ]
)


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


@dataclass(slots=True)
class TradeWriter:
    """Buffered parquet writer that flushes per-day shards.

    Not thread-safe; create one writer per producer task.
    """

    data_dir: Path
    flush_every: int = 1000
    _buffer: list[Trade] = field(default_factory=list)

    def append(self, trade: Trade) -> None:
        self._buffer.append(trade)
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        # Group by (exchange, date) so we write each shard once.
        groups: dict[tuple[str, str], list[Trade]] = {}
        for tr in self._buffer:
            day = datetime.fromtimestamp(tr.timestamp_ms / 1000.0, tz=UTC).strftime("%Y-%m-%d")
            groups.setdefault((tr.exchange, day), []).append(tr)

        for (exchange, day), trades in groups.items():
            shard_dir = self.data_dir / "trades" / exchange
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard_path = shard_dir / f"{day}.parquet"
            table = _trades_to_table(trades)
            if shard_path.exists():
                existing = pq.read_table(shard_path)
                table = pa.concat_tables([existing, table])
            pq.write_table(table, shard_path, compression="zstd")
        self._buffer.clear()

    def close(self) -> None:
        self.flush()


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


def load_trades(
    data_dir: Path,
    *,
    exchange: str | None = None,
    symbol: str | None = None,
) -> pd.DataFrame:
    """Load trades from parquet into a pandas DataFrame.

    Returns an empty DataFrame with the correct columns if nothing has been
    collected yet.
    """

    glob = parquet_glob(data_dir)
    cols = ["timestamp_ms", "exchange", "symbol", "price", "amount", "side", "trade_id"]
    try:
        with duckdb.connect() as con:
            query = f"SELECT * FROM read_parquet('{glob}', hive_partitioning=0)"
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
            df = con.execute(query, params).df()
    except duckdb.IOException:
        return pd.DataFrame(columns=cols)

    if df.empty:
        return pd.DataFrame(columns=cols)
    return df
