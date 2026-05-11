---
name: lead-lag-scanner-testing
description: Run lint/test/typecheck and smoke-test the collect+web+analyze pipeline. Use when validating changes to the lead-lag-scanner repo end-to-end before opening or updating a PR.
---

# Testing the lead-lag-scanner pipeline

This project ships four CLI commands that work on the same parquet shards under `data/trades/<exchange>/<YYYY-MM-DD>-<seq>.parquet`:

- `lead-lag-scanner collect` — async public-trade collector (ccxt + ccxt.pro)
- `lead-lag-scanner analyze` — cross-correlation + bootstrap CI
- `lead-lag-scanner report` — markdown + JSON
- `lead-lag-scanner web` — live FastAPI dashboard (Russian UI)
- `lead-lag-scanner dashboard` — terminal Rich UI (alternative to `web`)

## 0. One-time per session

```bash
cd /home/ubuntu/repos/lead-lag-scanner
uv sync
uv sync --extra ws    # ccxt.pro is gated behind the `ws` extra
```

## 1. Static checks (always run before pushing)

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run pyright src/
uv run pytest tests/ -q
```

Notes:
- `pyright` currently reports 8 pre-existing errors in `tests/test_web.py` related to pandas `itertuples()` typing. Treat them as a known baseline; only escalate if your diff changed that file or the count grew.
- The full suite is fast (≤ 5 s); always run it, do not cherry-pick.

## 2. Smoke-test `collect` (90 s, single reachable exchange set)

```bash
uv run lead-lag-scanner run --duration 90
```

This is the all-in-one (`collect` + `analyze` + `report`) entry point and is safe to run on the Devin VM. Expect:
- `data/trades/<exchange>/<YYYY-MM-DD>-<seq>.parquet` files appearing as flushes happen
- `reports/report.md` and `reports/leaders.json` at the end
- 0 hard errors; *some* per-exchange WARN lines are normal (geo-blocks, ws disconnects)

## 3. Smoke-test the live `web` dashboard

```bash
# T1: start collector in the background
nohup uv run lead-lag-scanner collect --duration 600 > /tmp/collect.log 2>&1 &

# T2: start the dashboard
uv run lead-lag-scanner web --refresh 5 &
sleep 5
curl -fsS http://127.0.0.1:8000/api/state | head

# verify pairs grow
curl -fsS "http://127.0.0.1:8000/api/pairs?min_corr=0.2&min_obs=30&limit=50" | head
```

Green-light criteria:
- `n_trades` is increasing across successive `/api/state` calls
- `n_pairs` reaches > 0 within ~30–60 s once thresholds are loosened (`min_obs=30`, `min_corr=0.2`)
- `/tmp/collect.log` contains **no repeated** `load_trades failed: ... no magic bytes` lines and **no repeated** ccxt tracebacks (per-error blacklist + quiet-handler should keep it clean)

To tear down between iterations:
```bash
pkill -9 -f "lead-lag-scanner collect"
pkill -9 -f "lead-lag-scanner web"
sleep 1
```

## 4. Probe the symbol/exchange registry

```bash
uv run lead-lag-scanner probe-symbols --min-exchanges 5 --top 200 --out /tmp/probed-symbols.yaml
head /tmp/probed-symbols.yaml
```

Use this when changing `config/default.yaml` to confirm the listing of a symbol on enough exchanges.

## 5. Known geo-blocks / quirks (do not flag as bugs)

- **binance / bybit / mexc**: typically return HTTP 451/403 from Devin VMs. The collector logs a single WARN and skips them.
- **bitfinex**: REST rate-limits aggressively at default settings; expect occasional `RateLimitExceeded`.
- **mexc REST `fetch_trades`**: requires `until` when `since` is set; prefer `ccxt.pro` WebSocket.
- **upbit / cex / luno / kraken**: hit the permanent-error blacklist (`requires apikey`, `one symbol per instance`, ws keepalive close). After one INFO log they stop retrying — this is intentional.
- **htx (`AttributeError: 'Client' object has no attribute 'reset'`)**: ccxt.pro 4.4.x bug. Suppressed by the quiet-handler's traceback scan in `collector.py::install_quiet_loop_handler`.

## 6. Concurrency rules

- Multiple **collect** instances against the same `data/` are unsafe — always `pkill -9 -f "lead-lag-scanner collect"` before starting a new one.
- **collect + web** concurrently is safe (per-flush immutable parquet shards, atomic rename).
- The web dashboard's runtime-symbols persistence lives at `data/runtime-symbols.yaml`. An active `collect` will *not* pick up new tickers mid-run — restart the collector after adding tickers via the UI.

## 7. When you cannot reproduce a user-reported error

Ask for the last ≈30 lines of `collect.log` and `web` terminal output. The two most common categories on the requester's Mac:

1. *Many `watch_trades error` lines from one exchange* → add the exchange's specific error string to `_PERMANENT_ERROR_PATTERNS` in `src/lead_lag_scanner/collector.py`.
2. *`load_trades failed` in the web terminal* → verify shards are sequence-suffixed (`ls data/trades/<exchange>/ | head`); if you see only `<day>.parquet` files, the user is still on a pre-fix commit and needs `git pull && uv sync`.
