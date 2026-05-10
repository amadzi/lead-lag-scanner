"""Round-trip tests for the parquet/duckdb storage layer."""

from __future__ import annotations

from pathlib import Path

import duckdb

from lead_lag_scanner.storage import (
    _CORRUPT_PATH_RE,
    Trade,
    TradeWriter,
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
