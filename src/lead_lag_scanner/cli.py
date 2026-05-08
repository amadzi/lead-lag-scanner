"""Command-line entry point.

Usage::

    lead-lag-scanner collect [--config PATH] [--duration SECONDS]
    lead-lag-scanner analyze [--config PATH]
    lead-lag-scanner report  [--config PATH]
    lead-lag-scanner run     [--config PATH] [--duration SECONDS]   # all three
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import replace
from pathlib import Path

import click
import structlog

from . import __version__
from .analyzer import analyze_all
from .collector import collect
from .config import Config, load_config
from .reporter import write_reports
from .storage import build_duckdb_view, load_trades

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
    results = analyze_all(trades, config.analyzer)
    md_path, json_path = write_reports(results, config.report)
    click.echo(f"wrote {md_path} and {json_path}", err=True)
    click.echo(f"reported {len(results)} pairs above min_obs threshold", err=True)


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
    results = analyze_all(trades, config.analyzer)
    md_path, json_path = write_reports(results, config.report)
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
    results = analyze_all(trades, config.analyzer)
    write_reports(results, config.report)
    click.echo("run complete", err=True)


if __name__ == "__main__":  # pragma: no cover
    cli()
