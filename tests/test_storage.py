"""Round-trip tests for the parquet/duckdb storage layer."""

from __future__ import annotations

from pathlib import Path

import duckdb

from lead_lag_scanner.storage import (
    Trade,
    TradeWriter,
    build_duckdb_view,
    load_trades,
)


def _make_trade(ts_ms: int, exchange: str = "binance", symbol: str = "BTC/USDT") -> Trade:
    return Trade(
        timestamp_ms=ts_ms,
        exchange=exchange,
        symbol=symbol,
        price=50_000.0 + ts_ms % 10,
        amount=0.01,
        side="buy",
        trade_id=f"t-{ts_ms}",
    )


def test_writer_persists_and_load_returns_dataframe(tmp_path: Path) -> None:
    writer = TradeWriter(data_dir=tmp_path, flush_every=2)
    base_ts = 1_700_000_000_000  # 2023-11-14 ish
    trades = [_make_trade(base_ts + i * 1000) for i in range(5)]
    for t in trades:
        writer.append(t)
    writer.close()

    df = load_trades(tmp_path)
    assert len(df) == 5
    assert set(df["exchange"].unique()) == {"binance"}
    assert set(df["symbol"].unique()) == {"BTC/USDT"}


def test_writer_partitions_by_day(tmp_path: Path) -> None:
    writer = TradeWriter(data_dir=tmp_path, flush_every=10)
    day1_ts = 1_700_000_000_000
    day2_ts = day1_ts + 24 * 3600 * 1000
    writer.append(_make_trade(day1_ts))
    writer.append(_make_trade(day2_ts))
    writer.close()

    shard_files = sorted((tmp_path / "trades" / "binance").glob("*.parquet"))
    assert len(shard_files) == 2


def test_load_returns_empty_when_no_data(tmp_path: Path) -> None:
    df = load_trades(tmp_path)
    assert df.empty
    expected_cols = {
        "timestamp_ms",
        "exchange",
        "symbol",
        "price",
        "amount",
        "side",
        "trade_id",
    }
    assert expected_cols.issubset(df.columns)


def test_build_duckdb_view(tmp_path: Path) -> None:
    writer = TradeWriter(data_dir=tmp_path, flush_every=10)
    writer.append(_make_trade(1_700_000_000_000))
    writer.append(_make_trade(1_700_000_001_000, exchange="okx"))
    writer.close()

    duckdb_path = tmp_path / "trades.duckdb"
    build_duckdb_view(tmp_path, duckdb_path)
    assert duckdb_path.exists()

    with duckdb.connect(str(duckdb_path)) as con:
        n = con.execute("SELECT COUNT(*) FROM trades").fetchone()
        assert n is not None
        assert n[0] == 2
