# lead-lag-scanner

Discover **lead / follower** relationships between crypto spot exchanges — i.e. find
pairs where one exchange's price systematically moves *before* another's. Such
pairs are the foundation of **latency-arbitrage** and **lead-lag** statistical
strategies.

The tool is split into three independent stages so you can re-run any of them:

1. **`collect`** — stream public trades over WebSocket / REST from a configurable
   list of exchanges and persist them to columnar storage (Parquet + DuckDB).
2. **`analyze`** — for every (symbol, exchange-pair) combination, compute the
   cross-correlation of returns at a grid of lags, locate the lag that maximises
   correlation, and label the leader / follower.
3. **`report`** — produce a Markdown summary plus a machine-readable
   `leaders.json` config that downstream execution bots can consume.

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

# 6. Collect 1 hour of public trades from the default exchange set (~28 exchanges)
uv run lead-lag-scanner collect

# 7. Analyze + emit reports
uv run lead-lag-scanner analyze
uv run lead-lag-scanner report

# Or do all three in one shot:
uv run lead-lag-scanner run
```

After the run, look at:
- `reports/report.md` — human-readable Markdown table sorted by `|corr|`.
- `reports/leaders.json` — machine-readable, one entry per pair.
- `data/trades/<exchange>/<YYYY-MM-DD>.parquet` — raw tick data, re-analyzable.

### Tuning for your run

The default config (`config/default.yaml`) is reasonable but you'll likely
want to override:

- **Duration** — defaults to 3600 s (1 hour). For a quick test, override at the
  CLI: `uv run lead-lag-scanner collect --duration 600` (10 min).
- **Symbols** — defaults to `BTC/USDT`, `ETH/USDT`, `SOL/USDT`. Edit
  `config/default.yaml` or pass your own `--config` YAML.
- **Exchanges** — 28 curated by default; some may be geo-blocked from your
  ISP (in which case the collector logs a warning and skips them). To run on
  a smaller set, copy `config/default.yaml` and trim the list.

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
tests/
    test_analyzer.py # property + golden-data tests on synthetic series
    test_storage.py
    test_config.py
```

## License

MIT
