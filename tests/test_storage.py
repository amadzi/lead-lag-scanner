"""Round-trip tests for the parquet/duckdb storage layer."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from lead_lag_scanner import storage as _storage_module
from lead_lag_scanner.storage import (
    _CORRUPT_PATH_RE,
    _DUCKDB_FAILURE_BACKOFF_SECONDS,
    Trade,
    TradeWriter,
    _clear_duckdb_skip,
    _disable_duckdb_until_retry,
    _duckdb_should_skip,
    _handle_generic_duckdb_failure,
    _identify_offending_shards,
    _load_trades_via_pyarrow,
    _promote_to_trade_schema,
    _quarantine_corrupt_shard_from_error,
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


# ---------------------------------------------------------------------------
# Corrupt-shard recovery (legacy <YYYY-MM-DD>.parquet files written in-place
# by a previous version of the writer that was interrupted mid-write).
# ---------------------------------------------------------------------------


def test_corrupt_path_re_extracts_path_from_duckdb_message() -> None:
    """Regex must pull the file path out of DuckDB's actual error string."""

    msg = (
        "Invalid Input Error: No magic bytes found at end of file "
        "'data/trades/gate/2026-05-09.parquet'"
    )
    match = _CORRUPT_PATH_RE.search(msg)
    assert match is not None
    assert match.group(1) == "data/trades/gate/2026-05-09.parquet"


def test_corrupt_path_re_handles_double_quotes() -> None:
    """Some DuckDB versions wrap paths in double quotes."""

    msg = 'Invalid Input Error: bad footer at end of file "/abs/path/x.parquet"'
    match = _CORRUPT_PATH_RE.search(msg)
    assert match is not None
    assert match.group(1) == "/abs/path/x.parquet"


def test_quarantine_corrupt_shard_renames_legacy_file(tmp_path: Path) -> None:
    """Given a real corrupt parquet on disk, the helper renames it to .corrupt."""

    bad = tmp_path / "trades" / "gate" / "2026-05-09.parquet"
    bad.parent.mkdir(parents=True, exist_ok=True)
    # Write 4 bytes — too short to be a valid parquet, footer-detection will
    # fail. We don't actually need DuckDB to reject this in the unit test;
    # we only care that the helper renames it on a synthesised error message.
    bad.write_bytes(b"PAR1")
    exc = duckdb.InvalidInputException(f"No magic bytes found at end of file '{bad.as_posix()}'")
    assert _quarantine_corrupt_shard_from_error(tmp_path, exc) is True
    assert not bad.exists()
    assert (tmp_path / "trades" / "gate" / "2026-05-09.parquet.corrupt").exists()


def test_load_trades_quarantines_corrupt_shard_and_returns_good_data(
    tmp_path: Path,
) -> None:
    """End-to-end: a good shard + a corrupt one yields good data on the second pass.

    This is the actual user-facing scenario from the bug report — a legacy
    ``2026-05-09.parquet`` left over from a previous interrupted run was
    crashing the dashboard's read because ``except duckdb.IOException`` did
    not catch the ``InvalidInputException`` raised on the missing footer.
    """

    # Good shard via the normal writer path.
    writer = TradeWriter(data_dir=tmp_path, flush_every=10)
    writer.append(_make_trade(1_700_000_000_000))
    writer.append(_make_trade(1_700_000_001_000, exchange="okx"))
    writer.close()

    # Corrupt legacy shard — same exchange dir, no -<seq> suffix.
    bad = tmp_path / "trades" / "binance" / "2023-11-14.parquet"
    bad.write_bytes(b"PAR1\x00\x00\x00\x00")  # not a valid parquet footer

    df = load_trades(tmp_path)

    # Corrupt file moved aside; good rows still loaded.
    assert not bad.exists(), "legacy corrupt shard should be quarantined"
    assert (tmp_path / "trades" / "binance" / "2023-11-14.parquet.corrupt").exists()
    assert len(df) == 2
    assert set(df["exchange"].unique()) == {"binance", "okx"}


def test_load_trades_returns_empty_if_quarantine_fails(tmp_path: Path) -> None:
    """If the error message has no extractable path, return empty without looping."""

    # Create a directory that will look like a parquet to DuckDB but isn't —
    # DuckDB's error message for a directory-as-file does not contain the
    # path in a quoted form, so the regex won't match. We just verify the
    # function terminates instead of looping.
    fake_shard = tmp_path / "trades" / "binance" / "weird.parquet"
    fake_shard.parent.mkdir(parents=True, exist_ok=True)
    fake_shard.write_bytes(b"not a parquet")

    df = load_trades(tmp_path)
    # Either the file was quarantined (modern DuckDB versions surface the
    # path) or load returned empty (older versions). Both are correct
    # behaviour — what matters is that we did not deadlock.
    assert df.empty


# ---------------------------------------------------------------------------
# Per-shard pyarrow fallback (kicks in when DuckDB cannot map a column type
# to numpy/pandas — e.g. duckdb.NotImplementedException "don't know what
# type" — which is *not* a "corrupt file" condition. The fallback reads each
# parquet individually via pyarrow and converts via pa.Table.to_pandas.)
# ---------------------------------------------------------------------------


def test_promote_to_trade_schema_backfills_missing_columns() -> None:
    """A legacy schema missing local_recv_ts_ns is filled with zeros."""

    legacy = pa.table(
        {
            "timestamp_ms": pa.array([1, 2], type=pa.int64()),
            "exchange": pa.array(["x", "y"], type=pa.string()),
            "symbol": pa.array(["BTC", "ETH"], type=pa.string()),
            "price": pa.array([100.0, 200.0], type=pa.float64()),
            "amount": pa.array([0.1, 0.2], type=pa.float64()),
            "side": pa.array(["buy", "sell"], type=pa.string()),
            "trade_id": pa.array(["t1", "t2"], type=pa.string()),
        }
    )
    promoted = _promote_to_trade_schema(legacy)
    assert "local_recv_ts_ns" in promoted.column_names
    assert promoted["local_recv_ts_ns"].to_pylist() == [0, 0]


def test_promote_to_trade_schema_casts_int32_to_int64() -> None:
    """A shard with timestamp_ms as int32 is cast to int64."""

    weird = pa.table(
        {
            "timestamp_ms": pa.array([1, 2], type=pa.int32()),  # not int64
            "exchange": pa.array(["x", "y"], type=pa.string()),
            "symbol": pa.array(["BTC", "ETH"], type=pa.string()),
            "price": pa.array([100.0, 200.0], type=pa.float64()),
            "amount": pa.array([0.1, 0.2], type=pa.float64()),
            "side": pa.array(["buy", "sell"], type=pa.string()),
            "trade_id": pa.array(["t1", "t2"], type=pa.string()),
            "local_recv_ts_ns": pa.array([10, 20], type=pa.int64()),
        }
    )
    promoted = _promote_to_trade_schema(weird)
    assert promoted.schema.field("timestamp_ms").type == pa.int64()
    assert promoted["timestamp_ms"].to_pylist() == [1, 2]


def test_load_trades_via_pyarrow_reads_real_shards(tmp_path: Path) -> None:
    """The fallback path reads good shards and skips genuinely broken ones."""

    writer = TradeWriter(data_dir=tmp_path, flush_every=10)
    writer.append(_make_trade(1_700_000_000_000))
    writer.append(_make_trade(1_700_000_001_000, exchange="okx"))
    writer.close()

    # Genuinely corrupt file pyarrow cannot read either — should be quarantined.
    bad = tmp_path / "trades" / "kraken" / "2023-11-14.parquet"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"not parquet at all")

    df = _load_trades_via_pyarrow(tmp_path)
    assert len(df) == 2
    assert set(df["exchange"].unique()) == {"binance", "okx"}
    assert not bad.exists(), "pyarrow-unreadable shard should be quarantined"
    assert (tmp_path / "trades" / "kraken" / "2023-11-14.parquet.corrupt").exists()


def test_load_trades_via_pyarrow_filters_by_exchange(tmp_path: Path) -> None:
    writer = TradeWriter(data_dir=tmp_path, flush_every=10)
    writer.append(_make_trade(1_700_000_000_000, exchange="binance"))
    writer.append(_make_trade(1_700_000_001_000, exchange="okx"))
    writer.close()

    df = _load_trades_via_pyarrow(tmp_path, exchange="okx")
    assert len(df) == 1
    assert df["exchange"].iloc[0] == "okx"


# ---------------------------------------------------------------------------
# Generic-duckdb-failure recovery path: when DuckDB's read_parquet raises
# something other than IOException/InvalidInputException (e.g. a
# TProtocolException on a footer it can't parse but pyarrow can), we should
# (a) identify the offending shard, (b) quarantine it if pyarrow also can't
# read it, (c) cache the "use pyarrow" decision for a few minutes so the
# log isn't spammed on every dashboard refresh.
# ---------------------------------------------------------------------------


def test_handle_generic_failure_quarantines_when_pyarrow_also_fails(
    tmp_path: Path,
) -> None:
    """A shard neither DuckDB nor pyarrow can read is moved to *.parquet.corrupt."""

    writer = TradeWriter(data_dir=tmp_path, flush_every=10)
    writer.append(_make_trade(1_700_000_000_000))
    writer.close()

    bad = tmp_path / "trades" / "kraken" / "2023-11-14.parquet"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"not parquet at all")

    _clear_duckdb_skip()
    fake_exc = duckdb.Error("TProtocolException: Invalid data")
    df = _handle_generic_duckdb_failure(tmp_path, fake_exc, exchange=None, symbol=None)

    assert not bad.exists()
    assert (tmp_path / "trades" / "kraken" / "2023-11-14.parquet.corrupt").exists()
    # Good shard still loads via the pyarrow fallback.
    assert len(df) == 1
    _clear_duckdb_skip()


def test_handle_generic_failure_caches_pyarrow_decision_when_shard_is_duckdb_only_bad(
    tmp_path: Path, monkeypatch: object
) -> None:
    """A shard pyarrow can read but DuckDB cannot triggers a cached skip."""

    writer = TradeWriter(data_dir=tmp_path, flush_every=10)
    writer.append(_make_trade(1_700_000_000_000))
    writer.close()

    only_duckdb_fake_path = tmp_path / "trades" / "binance" / "2023-11-14-99999999.parquet"

    monkeypatch.setattr(  # type: ignore[attr-defined]
        _storage_module,
        "_identify_offending_shards",
        lambda _data_dir: [only_duckdb_fake_path],
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        _storage_module,
        "_pyarrow_can_read",
        lambda _shard: True,
    )

    _clear_duckdb_skip()
    fake_exc = duckdb.Error("TProtocolException: Invalid data")
    _handle_generic_duckdb_failure(tmp_path, fake_exc, exchange=None, symbol=None)

    # Skip cached so a re-entry within the backoff window short-circuits.
    assert _duckdb_should_skip() is True
    assert _duckdb_should_skip(now=1.0 + _DUCKDB_FAILURE_BACKOFF_SECONDS * 1e6) is False
    _clear_duckdb_skip()


def test_identify_offending_shards_returns_only_broken(tmp_path: Path) -> None:
    """The probe touches every shard and only flags the truly broken ones."""

    writer = TradeWriter(data_dir=tmp_path, flush_every=10)
    writer.append(_make_trade(1_700_000_000_000))
    writer.append(_make_trade(1_700_000_001_000, exchange="okx"))
    writer.close()
    bad = tmp_path / "trades" / "kraken" / "2023-11-14.parquet"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"not a parquet file")

    offenders = _identify_offending_shards(tmp_path)
    assert offenders == [bad]


def test_disable_then_clear_duckdb_skip_round_trip() -> None:
    _clear_duckdb_skip()
    assert _duckdb_should_skip() is False
    _disable_duckdb_until_retry("test reason")
    assert _duckdb_should_skip() is True
    _clear_duckdb_skip()
    assert _duckdb_should_skip() is False


def test_load_trades_via_pyarrow_handles_legacy_schema(tmp_path: Path) -> None:
    """A legacy shard without local_recv_ts_ns is loaded with zero filled in."""

    shard_dir = tmp_path / "trades" / "kucoin"
    shard_dir.mkdir(parents=True)
    legacy = pa.table(
        {
            "timestamp_ms": pa.array([1, 2], type=pa.int64()),
            "exchange": pa.array(["kucoin", "kucoin"], type=pa.string()),
            "symbol": pa.array(["BTC/USDT", "ETH/USDT"], type=pa.string()),
            "price": pa.array([100.0, 200.0], type=pa.float64()),
            "amount": pa.array([0.1, 0.2], type=pa.float64()),
            "side": pa.array(["buy", "sell"], type=pa.string()),
            "trade_id": pa.array(["t1", "t2"], type=pa.string()),
        }
    )
    pq.write_table(legacy, shard_dir / "2023-11-14.parquet")

    df = _load_trades_via_pyarrow(tmp_path)
    assert len(df) == 2
    assert "local_recv_ts_ns" in df.columns
    assert df["local_recv_ts_ns"].tolist() == [0, 0]
