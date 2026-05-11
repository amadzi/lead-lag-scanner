"""Command-line entry point.

Usage::

    lead-lag-scanner collect       [--config PATH] [--duration SECONDS]
    lead-lag-scanner analyze       [--config PATH]
    lead-lag-scanner report        [--config PATH]
    lead-lag-scanner dashboard     [--config PATH] [--top N] [--sort KEY] [--strict]
    lead-lag-scanner run           [--config PATH] [--duration SECONDS]
    lead-lag-scanner probe-symbols [--config PATH] [--min-exchanges N] [--top N]
                                   [--quote QUOTE] [--out PATH]
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import replace
from pathlib import Path

import click
import structlog
import uvicorn

from . import __version__
from .analyzer import analyze_all_with_diagnostics
from .collector import collect
from .config import Config, load_config
from .dashboard import run_dashboard
from .exchanges import probe_usdt_symbols
from .reporter import write_reports
from .storage import build_duckdb_view, load_trades
from .web import create_app

_LOG_LEVEL_ENV = "LEAD_LAG_SCANNER_LOG_LEVEL"


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(level),
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.dev.ConsoleRenderer(),
        ],
    )


def _override_duration(config: Config, duration: float | None) -> Config:
    if duration is None:
        return config
    return replace(config, collector=replace(config.collector, duration=duration))


@click.group()
@click.version_option(__version__, prog_name="lead-lag-scanner")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool) -> None:
    """lead-lag-scanner — discover lead/follower exchange pairs."""

    _configure_logging(verbose)


def _common_config_option() -> click.Option:
    return click.Option(
        ["--config", "config_path"],
        type=click.Path(dir_okay=False, path_type=Path),
        default=None,
        help="Path to YAML config (defaults to bundled config/default.yaml).",
    )


@cli.command("collect", help="Stream public trades into local parquet shards.")
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
)
@click.option(
    "--duration",
    type=float,
    default=None,
    help="Override collector.duration in seconds.",
)
def collect_cmd(config_path: Path | None, duration: float | None) -> None:
    config = _override_duration(load_config(config_path), duration)
    click.echo(
        f"collecting from {len(config.exchanges)} exchanges × "
        f"{len(config.symbols)} symbols for {config.collector.duration:.0f}s",
        err=True,
    )
    stats = asyncio.run(collect(config))
    click.echo(
        f"done: trades_received={stats.trades_received} "
        f"trades_written={stats.trades_written} errors={stats.errors}",
        err=True,
    )


@cli.command("analyze", help="Compute lead-lag results from collected parquet shards.")
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
)
def analyze_cmd(config_path: Path | None) -> None:
    config = load_config(config_path)
    trades = load_trades(config.storage.data_dir)
    if trades.empty:
        click.echo("no trades found in data/ — run `collect` first", err=True)
        sys.exit(1)
    click.echo(f"loaded {len(trades)} trades; running analyzer", err=True)
    output = analyze_all_with_diagnostics(trades, config.analyzer)
    md_path, json_path = write_reports(
        output.results, config.report, diagnostics=output.diagnostics
    )
    click.echo(f"wrote {md_path} and {json_path}", err=True)
    click.echo(
        f"reported {len(output.results)} pairs above min_obs threshold "
        f"(diagnostics for {len(output.diagnostics)} exchanges)",
        err=True,
    )


@cli.command("report", help="Re-emit Markdown + JSON reports without re-analyzing.")
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
)
def report_cmd(config_path: Path | None) -> None:
    """Re-running ``analyze`` already writes the reports; this command is kept
    as a convenience that re-derives them from the same source data so users
    can tweak ``report`` config without re-running everything.
    """

    config = load_config(config_path)
    trades = load_trades(config.storage.data_dir)
    output = analyze_all_with_diagnostics(trades, config.analyzer)
    md_path, json_path = write_reports(
        output.results, config.report, diagnostics=output.diagnostics
    )
    click.echo(f"wrote {md_path} and {json_path}", err=True)


@cli.command("build-duckdb", help="Build a DuckDB view over the parquet shards.")
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
)
def build_duckdb_cmd(config_path: Path | None) -> None:
    config = load_config(config_path)
    build_duckdb_view(config.storage.data_dir, config.storage.duckdb_path)
    click.echo(f"created {config.storage.duckdb_path}", err=True)


@cli.command(
    "dashboard",
    help="Live terminal dashboard: re-reads trades and re-ranks pairs every N seconds.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
)
@click.option(
    "--refresh",
    type=float,
    default=None,
    help="Seconds between refreshes (overrides dashboard.refresh_seconds).",
)
@click.option(
    "--top",
    "top_k",
    type=int,
    default=None,
    help="Number of pairs to display (overrides dashboard.top_k).",
)
@click.option(
    "--sort",
    "sort_by",
    type=click.Choice(["edge_bps", "abs_corr", "n_obs", "lag_abs"]),
    default=None,
    help="Sort column (overrides dashboard.sort_by).",
)
@click.option(
    "--only-directional",
    is_flag=True,
    default=False,
    help="Only show pairs with a non-zero leader/follower lag.",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Use the report-grade min_obs / min_abs_correlation instead of the dashboard defaults.",
)
def dashboard_cmd(
    config_path: Path | None,
    refresh: float | None,
    top_k: int | None,
    sort_by: str | None,
    only_directional: bool,
    strict: bool,
) -> None:
    config = load_config(config_path)
    dash = config.dashboard
    if refresh is not None:
        dash = replace(dash, refresh_seconds=refresh)
    if top_k is not None:
        dash = replace(dash, top_k=top_k)
    if sort_by is not None:
        dash = replace(dash, sort_by=sort_by)
    if only_directional:
        dash = replace(dash, only_directional=True)
    if strict:
        dash = replace(
            dash,
            min_obs=config.analyzer.min_obs,
            min_abs_correlation=config.report.min_abs_correlation,
            bootstrap_n_iter=config.analyzer.bootstrap.n_iter,
        )
    run_dashboard(replace(config, dashboard=dash))


@cli.command(
    "web",
    help="Start the in-browser dashboard (Russian UI, filters, manual ticker add).",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
)
@click.option(
    "--host",
    type=str,
    default="127.0.0.1",
    show_default=True,
    help="Host/interface to bind to. Use 0.0.0.0 to expose on the LAN.",
)
@click.option(
    "--port",
    type=int,
    default=8000,
    show_default=True,
    help="TCP port to listen on.",
)
@click.option(
    "--refresh",
    type=float,
    default=None,
    help="Seconds between background re-analysis cycles (overrides dashboard.refresh_seconds).",
)
def web_cmd(
    config_path: Path | None,
    host: str,
    port: int,
    refresh: float | None,
) -> None:
    config = load_config(config_path)
    app = create_app(config, refresh_seconds=refresh)
    click.echo(
        f"web dashboard listening on http://{host}:{port}/  "
        f"(refresh={refresh or config.dashboard.refresh_seconds:.0f}s, "
        f"data_dir={config.storage.data_dir})",
        err=True,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")


@cli.command("run", help="collect → analyze → report in one shot.")
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
)
@click.option("--duration", type=float, default=None)
def run_cmd(config_path: Path | None, duration: float | None) -> None:
    config = _override_duration(load_config(config_path), duration)
    asyncio.run(collect(config))
    trades = load_trades(config.storage.data_dir)
    output = analyze_all_with_diagnostics(trades, config.analyzer)
    write_reports(output.results, config.report, diagnostics=output.diagnostics)
    click.echo("run complete", err=True)


@cli.command(
    "probe-symbols",
    help=(
        "Probe every exchange in the config (or all known ccxt exchanges) for "
        "USDT-quoted symbols and emit a YAML snippet of those listed on at "
        "least N exchanges, sorted by coverage."
    ),
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
)
@click.option(
    "--min-exchanges",
    type=int,
    default=5,
    show_default=True,
    help="Only emit symbols listed on at least this many exchanges.",
)
@click.option(
    "--top",
    "top_n",
    type=int,
    default=None,
    help="Cap the output to the top-N most-listed symbols.",
)
@click.option(
    "--quote",
    type=str,
    default="USDT",
    show_default=True,
    help="Quote currency to filter on.",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the YAML snippet to this file instead of stdout.",
)
def probe_symbols_cmd(
    config_path: Path | None,
    min_exchanges: int,
    top_n: int | None,
    quote: str,
    out_path: Path | None,
) -> None:
    config = load_config(config_path)
    click.echo(
        f"probing {len(config.exchanges)} exchanges for {quote}-quoted spot symbols…",
        err=True,
    )
    coverage = asyncio.run(probe_usdt_symbols(list(config.exchanges), quote=quote))
    eligible = [
        (sym, exchanges) for sym, exchanges in coverage.items() if len(exchanges) >= min_exchanges
    ]
    eligible.sort(key=lambda kv: (-len(kv[1]), kv[0]))
    if top_n is not None:
        eligible = eligible[:top_n]
    click.echo(
        f"  {len(eligible)} symbol(s) listed on ≥{min_exchanges} exchanges "
        f"(of {len(coverage)} total)",
        err=True,
    )

    lines: list[str] = [
        f"# probe-symbols: {len(eligible)} {quote}-quoted symbols on "
        f">={min_exchanges} of {len(config.exchanges)} configured exchanges",
        "symbols:",
    ]
    for sym, exchanges in eligible:
        lines.append(f"  - {sym}  # {len(exchanges)} exchanges: {', '.join(sorted(exchanges))}")
    snippet = "\n".join(lines) + "\n"
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(snippet, encoding="utf-8")
        click.echo(f"wrote {out_path}", err=True)
    else:
        click.echo(snippet)


if __name__ == "__main__":  # pragma: no cover
    cli()
