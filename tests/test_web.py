"""Tests for the FastAPI web dashboard."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from lead_lag_scanner.config import (
    DEFAULT_CONFIG_PATH,
    Config,
    load_config,
    load_runtime_symbols,
    merge_runtime_symbols,
    runtime_symbols_path,
    save_runtime_symbols,
)
from lead_lag_scanner.storage import Trade, TradeWriter
from lead_lag_scanner.web import (
    PairOut,
    _filter_pairs,
    _normalise_symbol,
    _sort_pairs,
    create_app,
)


@pytest.fixture
def base_config(tmp_path: Path) -> Config:
    """Default config with storage redirected to a temporary dir."""

    cfg = load_config(DEFAULT_CONFIG_PATH)
    storage = replace(cfg.storage, data_dir=tmp_path / "data")
    storage.data_dir.mkdir(parents=True, exist_ok=True)
    return replace(cfg, storage=storage)


@pytest.fixture
def trades_df() -> pd.DataFrame:
    """Synthetic two-exchange tick data with a +1s lag on `gate`."""

    base_ts = 1_700_000_000_000
    rows: list[dict[str, object]] = []
    for i in range(900):
        ts_ms = base_ts + i * 1000
        rows.append(
            dict(
                timestamp_ms=ts_ms,
                exchange="okx",
                symbol="BTC/USDT",
                price=100.0 + (i % 13) * 0.5,
                amount=1.0,
                side="buy",
                trade_id=f"o-{i}",
                local_recv_ts_ns=ts_ms * 1_000_000,
            )
        )
        rows.append(
            dict(
                timestamp_ms=ts_ms + 1_000,  # gate trades arrive 1s after okx
                exchange="gate",
                symbol="BTC/USDT",
                price=100.0 + (i % 13) * 0.5,
                amount=1.0,
                side="buy",
                trade_id=f"g-{i}",
                local_recv_ts_ns=(ts_ms + 1_000) * 1_000_000,
            )
        )
    return pd.DataFrame(rows)


def _write_trades(data_dir: Path, trades: pd.DataFrame) -> None:
    """Persist `trades` into the data_dir using the storage layout."""

    writer = TradeWriter(data_dir=data_dir)
    for r in trades.itertuples(index=False):
        writer.append(
            Trade(
                timestamp_ms=int(r.timestamp_ms),
                exchange=str(r.exchange),
                symbol=str(r.symbol),
                price=float(r.price),
                amount=float(r.amount),
                side=str(r.side),
                trade_id=str(r.trade_id),
                local_recv_ts_ns=int(r.local_recv_ts_ns),
            )
        )
    writer.close()


# ---------------------------------------------------------------------------
# Symbol validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("btc/usdt", "BTC/USDT"),
        ("  ETH/USDT  ", "ETH/USDT"),
        ("PEPE/USDT", "PEPE/USDT"),
        ("1000PEPE/USDT", "1000PEPE/USDT"),
        ("KAS-PERP/USDT", "KAS-PERP/USDT"),
    ],
)
def test_normalise_symbol_accepts_valid(raw: str, expected: str) -> None:
    try:
        assert _normalise_symbol(raw) == expected
    except HTTPException as exc:  # pragma: no cover - debugging aid
        pytest.fail(f"unexpectedly rejected {raw!r}: {exc.detail}")


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "BTC",
        "BTC/",
        "/USDT",
        "btc usdt",
        "БТС/USDT",  # Cyrillic
        "../etc/passwd",
        "<script>",
    ],
)
def test_normalise_symbol_rejects_invalid(raw: str) -> None:
    with pytest.raises(HTTPException):
        _normalise_symbol(raw)


# ---------------------------------------------------------------------------
# Runtime-symbols persistence (config helpers)
# ---------------------------------------------------------------------------


def test_runtime_symbols_round_trip(tmp_path: Path) -> None:
    assert load_runtime_symbols(tmp_path) == ()
    saved = save_runtime_symbols(tmp_path, ("PEPE/USDT", "WIF/USDT"))
    assert saved == runtime_symbols_path(tmp_path)
    assert load_runtime_symbols(tmp_path) == ("PEPE/USDT", "WIF/USDT")


def test_runtime_symbols_dedup(tmp_path: Path) -> None:
    save_runtime_symbols(tmp_path, ("PEPE/USDT", "PEPE/USDT", " ", "WIF/USDT"))
    # Duplicates are kept on disk by save (it just dumps), but load_runtime_symbols
    # filters duplicates / empty strings before returning.
    assert load_runtime_symbols(tmp_path) == ("PEPE/USDT", "WIF/USDT")


def test_runtime_symbols_malformed_yaml_returns_empty(tmp_path: Path) -> None:
    runtime_symbols_path(tmp_path).write_text("not: [valid yaml", encoding="utf-8")
    assert load_runtime_symbols(tmp_path) == ()


def test_merge_runtime_symbols_appends(base_config: Config) -> None:
    save_runtime_symbols(base_config.storage.data_dir, ("ZZZ/USDT", "BTC/USDT"))
    merged = merge_runtime_symbols(base_config)
    # BTC/USDT should not be duplicated, ZZZ/USDT should be appended at the end.
    assert "ZZZ/USDT" in merged.symbols
    assert merged.symbols.count("BTC/USDT") == 1
    # Original ordering preserved up to the existing length.
    assert merged.symbols[: len(base_config.symbols)] == base_config.symbols


# ---------------------------------------------------------------------------
# Filter / sort helpers
# ---------------------------------------------------------------------------


def _mk_pair(
    *,
    symbol: str = "BTC/USDT",
    a: str = "okx",
    b: str = "gate",
    leader: str = "okx",
    follower: str = "gate",
    edge: float = 5.0,
    corr: float = 0.5,
    n: int = 200,
    lag: float = 1.0,
) -> PairOut:
    return PairOut(
        symbol=symbol,
        exchange_a=a,
        exchange_b=b,
        leader=leader,
        follower=follower,
        best_lag_seconds=lag,
        best_correlation=corr,
        abs_correlation=abs(corr),
        correlation_at_zero=0.1,
        lag_ci_low_seconds=0.5,
        lag_ci_high_seconds=1.5,
        follower_return_std=0.001,
        edge_bps=edge,
        n_obs=n,
        n_co_active=n,
        active_rate_a=0.9,
        active_rate_b=0.9,
    )


def test_filter_pairs_min_corr() -> None:
    p1 = _mk_pair(corr=0.9)
    p2 = _mk_pair(corr=0.2, symbol="ETH/USDT")
    out = _filter_pairs(
        [p1, p2],
        exchanges=None,
        symbols=None,
        min_abs_corr=0.5,
        min_n_obs=0,
        only_directional=False,
    )
    assert out == [p1]


def test_filter_pairs_exchange_subset() -> None:
    p1 = _mk_pair(a="okx", b="gate")
    p2 = _mk_pair(a="binance", b="kucoin")
    out = _filter_pairs(
        [p1, p2],
        exchanges={"okx"},
        symbols=None,
        min_abs_corr=0.0,
        min_n_obs=0,
        only_directional=False,
    )
    assert out == [p1]


def test_filter_pairs_only_directional() -> None:
    p1 = _mk_pair(leader="okx", follower="gate")
    p2 = _mk_pair(leader="none", follower="none", lag=0.0)
    out = _filter_pairs(
        [p1, p2],
        exchanges=None,
        symbols=None,
        min_abs_corr=0.0,
        min_n_obs=0,
        only_directional=True,
    )
    assert out == [p1]


def test_sort_pairs_edge_desc() -> None:
    p1 = _mk_pair(edge=2.0)
    p2 = _mk_pair(edge=10.0)
    p3 = _mk_pair(edge=-3.0)
    out = _sort_pairs([p1, p2, p3], "edge_bps")
    assert [p.edge_bps for p in out] == [10.0, 2.0, -3.0]


def test_sort_pairs_unknown_falls_back_to_edge() -> None:
    p1 = _mk_pair(edge=2.0)
    p2 = _mk_pair(edge=10.0)
    out = _sort_pairs([p1, p2], "nonsense")
    assert out[0].edge_bps == 10.0


# ---------------------------------------------------------------------------
# HTTP endpoint smoke
# ---------------------------------------------------------------------------


def test_state_endpoint_empty(base_config: Config) -> None:
    app = create_app(base_config, refresh_seconds=999)
    with TestClient(app) as client:
        res = client.get("/api/state")
        assert res.status_code == 200
        body = res.json()
        assert body["n_trades"] == 0
        assert body["n_pairs"] == 0
        assert body["exchanges_configured"]
        assert body["symbols_configured"]
        assert body["runtime_symbols"] == []


def test_index_returns_html(base_config: Config) -> None:
    app = create_app(base_config, refresh_seconds=999)
    with TestClient(app) as client:
        res = client.get("/")
        assert res.status_code == 200
        assert "lead-lag-scanner" in res.text
        # Russian UI sanity
        assert "Фильтры" in res.text
        assert "Добавить тикер" in res.text


def test_pairs_endpoint_with_real_analysis(base_config: Config, trades_df: pd.DataFrame) -> None:
    _write_trades(base_config.storage.data_dir, trades_df)
    # Tighten the analyzer so the smoke fixture (15 minutes of data) qualifies.
    cfg = replace(
        base_config,
        analyzer=replace(base_config.analyzer, min_obs=10, min_active_rate=0.0),
        dashboard=replace(base_config.dashboard, min_obs=10, bootstrap_n_iter=20),
    )
    app = create_app(cfg, refresh_seconds=999)
    with TestClient(app) as client:
        res = client.get(
            "/api/pairs",
            params={"sort_by": "edge_bps", "min_corr": 0.0, "min_obs": 0, "limit": 50},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["n_total"] >= 1
        # Filtering by an unknown symbol returns zero pairs.
        res = client.get("/api/pairs", params={"symbols": "DOES-NOT/EXIST"})
        assert res.status_code == 200
        assert res.json()["n_filtered"] == 0


def test_add_and_delete_symbol(base_config: Config) -> None:
    # Use synthetic symbols that won't collide with the default config so we
    # exercise the "added" path rather than the "already in main config" path.
    new_a = "ZZZTEST1/USDT"
    new_b = "ZZZTEST2/USDT"
    assert new_a not in base_config.symbols
    assert new_b not in base_config.symbols

    app = create_app(base_config, refresh_seconds=999)
    with TestClient(app) as client:
        # Initial list is empty.
        assert client.get("/api/symbols").json()["runtime_symbols"] == []

        res = client.post("/api/symbols", json={"symbol": new_a.lower()})
        assert res.status_code == 200
        body = res.json()
        assert body["runtime_symbols"] == [new_a]
        assert new_a in body["message"]

        # Adding a second one preserves order.
        res = client.post("/api/symbols", json={"symbol": new_b})
        assert res.status_code == 200
        assert res.json()["runtime_symbols"] == [new_a, new_b]

        # Adding a duplicate is a no-op (no error, helpful message).
        res = client.post("/api/symbols", json={"symbol": new_a})
        assert res.status_code == 200
        assert res.json()["runtime_symbols"] == [new_a, new_b]

        # Adding a symbol that already exists in the main config is a no-op.
        already = base_config.symbols[0]
        res = client.post("/api/symbols", json={"symbol": already})
        assert res.status_code == 200
        assert res.json()["runtime_symbols"] == [new_a, new_b]

        # Bad symbol -> 400.
        res = client.post("/api/symbols", json={"symbol": "garbage"})
        assert res.status_code == 400

        # Deletion works and round-trips through the GET endpoint.
        res = client.delete(f"/api/symbols/{new_a.replace('/', '%2F')}")
        assert res.status_code == 200
        assert res.json()["runtime_symbols"] == [new_b]

        # Deleting an unknown symbol returns 404.
        res = client.delete("/api/symbols/DOES-NOT%2FEXIST")
        assert res.status_code == 404


def test_force_refresh(base_config: Config) -> None:
    app = create_app(base_config, refresh_seconds=999)
    with TestClient(app) as client:
        res = client.post("/api/refresh")
        assert res.status_code == 200
        body = res.json()
        assert body["ok"] is True
        # last_error may be None (no data) but the field must exist.
        assert "computed_at" in body
        assert "n_pairs" in body
