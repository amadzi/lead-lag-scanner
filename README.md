# lead-lag-scanner

Discover **lead / follower** relationships between crypto spot exchanges — i.e. find
pairs where one exchange's price systematically moves *before* another's. Such
pairs are the foundation of **latency-arbitrage** and **lead-lag** statistical
strategies.

The tool is split into four independent stages so you can re-run any of them:

1. **`collect`** — stream public trades over WebSocket / REST from a configurable
   list of exchanges and persist them to columnar storage (Parquet + DuckDB).
2. **`analyze`** — for every (symbol, exchange-pair) combination, compute the
   cross-correlation of returns at a grid of lags, locate the lag that maximises
   correlation, and label the leader / follower.
3. **`report`** — produce a Markdown summary plus a machine-readable
   `leaders.json` config that downstream execution bots can consume.
4. **`dashboard`** — live terminal UI that re-reads the parquet shards and
   re-ranks pairs by potential edge every N seconds, so you can watch
   leadership emerge while `collect` is still running.

> :warning: This repository is **research tooling** only. It does not place
> orders. Use the output as input to a separate execution engine (Rust / Go /
> Node / Python) that you control.

## Quick start (macOS — Mac 2022, M1/M2 or Intel)

```bash
# 1. Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install uv (fast Python package manager) and git
brew install uv git

# 3. Clone this repo
git clone https://github.com/amadzi/lead-lag-scanner.git
cd lead-lag-scanner

# 4. Install dependencies (creates a local .venv automatically)
uv sync

# 5. (optional) Add the WebSocket extra for ccxt.pro - faster, lower-latency
#    Without this you fall back to REST polling, which is fine but slower.
uv sync --extra ws

# 6. Collect 1 hour of public trades from the default 51 exchanges × ~28 symbols
uv run lead-lag-scanner collect

# 7. Analyze + emit reports
uv run lead-lag-scanner analyze
uv run lead-lag-scanner report

# Or do all three in one shot:
uv run lead-lag-scanner run

# 8. (recommended) Open the live dashboard in a SECOND terminal while
#    `collect` is running in the first. Refreshes every 30s, sorted by
#    potential edge in basis points.
uv run lead-lag-scanner dashboard
```

### Live dashboard

```bash
# default: top 30 pairs sorted by edge_bps, 30s refresh
uv run lead-lag-scanner dashboard

# only show pairs with directional leadership (non-zero lag)
uv run lead-lag-scanner dashboard --only-directional

# sort by something else
uv run lead-lag-scanner dashboard --sort abs_corr
uv run lead-lag-scanner dashboard --sort lag_abs
uv run lead-lag-scanner dashboard --sort n_obs

# tighten thresholds to report-grade (min_obs=600, min_abs_corr=0.4)
uv run lead-lag-scanner dashboard --strict

# faster refresh + more rows
uv run lead-lag-scanner dashboard --refresh 10 --top 50
```

The dashboard is read-only — it reuses the same parquet shards `collect`
writes, so you can leave it running (or open multiple windows with different
sort keys) while data is being recorded.

After the run, look at:
- `reports/report.md` — human-readable Markdown table sorted by `|corr|`.
- `reports/leaders.json` — machine-readable, one entry per pair.
- `data/trades/<exchange>/<YYYY-MM-DD>.parquet` — raw tick data, re-analyzable.

### Tuning for your run

The default config (`config/default.yaml`) is reasonable but you'll likely
want to override:

- **Duration** — defaults to 3600 s (1 hour). For a quick test, override at the
  CLI: `uv run lead-lag-scanner collect --duration 600` (10 min).
- **Symbols** — default set covers BTC / ETH / SOL plus high-volatility
  memecoins (DOGE, SHIB, PEPE, WIF, BONK, FLOKI, TRUMP, PUMP, VIRTUAL, PNUT,
  MOODENG, FARTCOIN, POPCAT, BRETT, BOME, TURBO, MEW, GOAT) and volatile
  alt-L1 / themed picks (SUI, APT, SEI, TIA, INJ, ORDI, WLD, AIXBT). Edit
  `config/default.yaml` or pass your own `--config` YAML to add or trim. The
  collector silently skips any (exchange, symbol) pair that is not listed,
  so adding a symbol that exists on only some venues is safe.
- **Exchanges** — 51 curated by default; some may be geo-blocked from your
  ISP (in which case the collector logs a warning and skips them). To run on
  a smaller set, copy `config/default.yaml` and trim the list.
- **Memecoins / shitcoins** — generally have *much* higher per-bar σ than BTC,
  so the `edge_bps` proxy can be larger even with weaker correlation. The
  trade-off is that liquidity is shallower and slippage matters more than for
  majors — treat any memecoin signal as input to a real fee + impact backtest,
  not as guaranteed PnL.

### Running in the background (e.g. overnight)

```bash
nohup uv run lead-lag-scanner collect --duration 86400 > collect.log 2>&1 &
echo $! > collect.pid

# Watch progress:
tail -f collect.log

# Stop early (clean shutdown - parquet files are flushed):
kill -TERM $(cat collect.pid)
```

### Troubleshooting

- **"skipping symbol - not listed on exchange"** — normal, that exchange
  doesn't carry that pair. Other exchanges still collect.
- **"ExchangeNotAvailable / 451 / 403"** — that exchange is geo-blocked from
  your IP. Ignore or remove from config. From a US IP, Binance/Bybit/MEXC
  are commonly blocked. From the EU you usually have access to all 28.
- **No pairs in report** — try lowering `report.min_abs_correlation` in your
  config (default 0.4) or `analyzer.min_obs` (default 600 = 10 min). For
  short collections (<10 min) reduce both.

The default configuration lives at `config/default.yaml`.

## Methodology

For each symbol shared between two exchanges *A* and *B*, the analyzer:

1. Resamples the trade stream into a 1-second grid of last-trade prices.
2. Computes log-returns on each side.
3. Computes Pearson correlation `corr(r_A(t), r_B(t + lag))` on a configurable
   lag grid (default ±5 s, step 0.1 s).
4. Picks the `lag*` that maximises the absolute correlation.
   - `lag* > 0`  ⇒  *A* leads *B* by `lag*` seconds.
   - `lag* < 0`  ⇒  *B* leads *A*.
5. Reports the maximum correlation, the lag, the number of overlapping
   observations, and a confidence band derived from a block-bootstrap.

A naive bps-edge estimate is also computed:

```
edge_bps ≈ |corr| × σ(returns_follower) × scaling − 2 × taker_fee_bps
```

where `scaling` is empirical and conservatively set so that only signals with
edge meaningfully above the round-trip fee are highlighted.

## Caveats

- Public trade feeds are *not* tick-true order-book updates; lead/lag estimates
  derived from them slightly under-count true microstructure lead.
- Apparent leadership often disappears once you account for fees, slippage, and
  market impact. The reporter intentionally flags pairs whose `edge_bps ≤ 0`.
- "Stable" leadership shifts over hours/days. Re-run periodically.

## Configuration

See `config/default.yaml` for the complete shape. Important knobs:

| Key                 | Meaning                                                     |
|---------------------|-------------------------------------------------------------|
| `exchanges`         | List of ccxt exchange ids to scan.                          |
| `symbols`           | Trading pairs to scan (must exist on every exchange listed).|
| `collector.duration`| How long (s) to record trades when running `collect`.       |
| `analyzer.lag_grid` | Lag grid (start, stop, step) in seconds.                    |
| `analyzer.min_obs`  | Minimum overlapping seconds before a pair is reported.      |
| `report.taker_bps`  | Round-trip taker fee assumption used for `edge_bps`.        |

## Project layout

```
src/lead_lag_scanner/
    cli.py           # click CLI entry point
    config.py        # typed configuration loader
    exchanges.py     # ccxt exchange factory + symbol normalisation
    collector.py     # async trade collector (ccxt async / ccxt.pro)
    storage.py       # Parquet + DuckDB persistence
    analyzer.py      # lead-lag computation
    reporter.py      # Markdown + JSON output
    dashboard.py     # live terminal dashboard (rich)
tests/
    test_analyzer.py # property + golden-data tests on synthetic series
    test_storage.py
    test_config.py
    test_reporter.py
    test_dashboard.py
```

## License

MIT
