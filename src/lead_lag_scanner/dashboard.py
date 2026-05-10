"""Live terminal dashboard.

Continuously reads the parquet shards under ``storage.data_dir`` and re-runs
the analyzer to surface the top-K (symbol, exchange-pair) lead-lag results
sorted by potential edge in basis points.

Designed to be run alongside ``collect``: launch the collector in one
terminal and ``dashboard`` in another. Both touch the same parquet shards,
but the dashboard only reads.

The :class:`DashboardConfig` defaults are intentionally looser than the
report's: ``min_obs=60`` and ``min_abs_correlation=0.3`` so the dashboard
shows preliminary signal a few minutes after collection starts. Use
``--strict`` (or override the config) to apply the report-grade thresholds.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime

import structlog
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .analyzer import LeadLagResult, analyze_all
from .config import AnalyzerConfig, BootstrapConfig, Config, DashboardConfig, ReportConfig
from .reporter import _edge_bps, _net_edge_bps, _tradeability
from .storage import load_trades

log = structlog.get_logger()


def _sort_key(sort_by: str, report_cfg: ReportConfig) -> Callable[[LeadLagResult], float]:
    """Return a key function ranking results so that *better* comes first.

    All keys return *negative* values because ``list.sort`` is ascending.
    """

    if sort_by == "tradeability":
        return lambda r: -_tradeability(r, report_cfg)
    if sort_by == "edge_bps":
        return lambda r: -_edge_bps(r, report_cfg.taker_bps)
    if sort_by == "net_edge_bps":
        return lambda r: -_net_edge_bps(r, report_cfg)
    if sort_by == "abs_corr":
        return lambda r: -abs(r.best_correlation)
    if sort_by == "n_obs":
        return lambda r: -float(r.n_obs)
    if sort_by == "lag_abs":
        return lambda r: -abs(r.best_lag_seconds)
    raise ValueError(f"unknown sort_by: {sort_by!r}")


def _filter_and_sort(
    results: list[LeadLagResult],
    dash: DashboardConfig,
    report_cfg: ReportConfig,
) -> list[LeadLagResult]:
    """Apply correlation / directional / sanity-flag filters and sort.

    The terminal dashboard hides ``boundary_lag`` and ``wide_ci`` by
    default — those are the two flag classes that almost always indicate
    an unstable estimate rather than a real lead-lag.
    """

    out = [r for r in results if abs(r.best_correlation) >= dash.min_abs_correlation]
    hide = dash.hide_flags
    if hide:
        out = [r for r in out if not any(f in hide for f in r.flags)]
    if dash.only_directional:
        out = [r for r in out if r.leader != "none"]
    out.sort(key=_sort_key(dash.sort_by, report_cfg))
    return out


def _ci_text(low: float, high: float) -> str:
    if math.isnan(low) or math.isnan(high):
        return "—"
    return f"[{low:+.1f}, {high:+.1f}]"


def _edge_style(edge: float, taker_bps: float) -> str:
    if edge > 0:
        return "bold green"
    if edge > -taker_bps:
        return "yellow"
    return "dim"


def _net_edge_style(net_edge: float) -> str:
    """Net-edge colour: green if it survives all costs, dim otherwise.

    There's no middle "yellow" tier here because once spread + slippage
    are netted out, a non-positive number is genuinely untradeable.
    """

    if net_edge > 0:
        return "bold green"
    return "dim"


def _flags_text(flags: tuple[str, ...]) -> Text:
    """Render sanity flags as a compact, colourised cell.

    Empty ⇒ "—" in dim. Any flag ⇒ comma-joined in red so they pop in
    the table even when the row otherwise looks attractive.
    """

    if not flags:
        return Text("—", style="dim")
    return Text(",".join(flags), style="red")


def render_table(
    results: list[LeadLagResult],
    report_cfg: ReportConfig,
    dash: DashboardConfig,
    *,
    n_trades: int | None = None,
    n_exchanges: int | None = None,
    n_symbols: int | None = None,
    cycle_seconds: float | None = None,
) -> Table:
    """Build a Rich table; pure function so it is unit-testable."""

    ranked = _filter_and_sort(results, dash, report_cfg)
    rows = ranked[: dash.top_k]

    table = Table(
        title=f"lead-lag-scanner — top {len(rows)} pairs by {dash.sort_by}",
        title_style="bold cyan",
        caption_style="dim",
        show_lines=False,
    )
    table.add_column("symbol", style="bold")
    table.add_column("leader", style="green")
    table.add_column("follower", style="red")
    table.add_column("lag (s)", justify="right")
    table.add_column("CI (s)", justify="center", style="dim")
    table.add_column("|corr|", justify="right")
    table.add_column("σ (bps)", justify="right", style="dim")
    table.add_column("edge (bps)", justify="right")
    table.add_column("net (bps)", justify="right", style="bold")
    table.add_column("score", justify="right")
    table.add_column("flags", justify="left")
    table.add_column("n", justify="right", style="dim")

    for r in rows:
        edge = _edge_bps(r, report_cfg.taker_bps)
        net = _net_edge_bps(r, report_cfg)
        score = _tradeability(r, report_cfg)
        leader = r.leader if r.leader != "none" else "—"
        follower = r.follower if r.follower != "none" else "—"
        sigma_bps = r.follower_return_std * 1e4
        table.add_row(
            r.symbol,
            leader,
            follower,
            f"{r.best_lag_seconds:+.1f}",
            _ci_text(r.lag_ci_low_seconds, r.lag_ci_high_seconds),
            f"{abs(r.best_correlation):.3f}",
            f"{sigma_bps:.1f}",
            Text(f"{edge:+.1f}", style=_edge_style(edge, report_cfg.taker_bps)),
            Text(f"{net:+.1f}", style=_net_edge_style(net)),
            f"{score:+.0f}",
            _flags_text(r.flags),
            f"{r.n_obs}",
        )

    meta_parts: list[str] = []
    if n_trades is not None:
        meta_parts.append(f"trades={n_trades:,}")
    if n_exchanges is not None:
        meta_parts.append(f"exchanges={n_exchanges}")
    if n_symbols is not None:
        meta_parts.append(f"symbols={n_symbols}")
    meta_parts.append(f"refresh={dash.refresh_seconds:.0f}s")
    if cycle_seconds is not None:
        meta_parts.append(f"cycle={cycle_seconds:.1f}s")
    meta_parts.append(datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"))
    table.caption = " · ".join(meta_parts)

    return table


def _effective_analyzer_config(base: AnalyzerConfig, dash: DashboardConfig) -> AnalyzerConfig:
    """Lower ``min_obs`` and ``bootstrap.n_iter`` so the dashboard is responsive."""

    new_bootstrap = BootstrapConfig(
        block_size=base.bootstrap.block_size,
        n_iter=dash.bootstrap_n_iter,
    )
    return replace(base, min_obs=dash.min_obs, bootstrap=new_bootstrap)


def _waiting_panel(message: str) -> Panel:
    return Panel(
        Text(message, style="yellow"),
        title="lead-lag-scanner — dashboard",
        border_style="cyan",
    )


def run_dashboard(config: Config) -> None:
    """Run the live dashboard until interrupted (Ctrl+C)."""

    console = Console()
    dash = config.dashboard
    eff_analyzer = _effective_analyzer_config(config.analyzer, dash)

    console.print(
        f"[cyan]dashboard[/cyan]: refresh={dash.refresh_seconds:.0f}s "
        f"top_k={dash.top_k} sort_by={dash.sort_by} "
        f"min_obs={dash.min_obs} min_abs_corr={dash.min_abs_correlation}"
    )
    console.print(f"[dim]reading from: {config.storage.data_dir}[/dim]")
    console.print("[dim]press Ctrl+C to exit[/dim]")

    try:
        with Live(console=console, screen=False, refresh_per_second=4) as live:
            while True:
                t_start = time.monotonic()
                try:
                    trades = load_trades(config.storage.data_dir)
                except Exception as exc:
                    log.warning("dashboard: load_trades failed", error=str(exc))
                    live.update(_waiting_panel(f"load_trades failed: {exc!s}"))
                    time.sleep(min(dash.refresh_seconds, 5.0))
                    continue

                if trades.empty:
                    live.update(
                        _waiting_panel(
                            "waiting for trades…\n"
                            "run `lead-lag-scanner collect` in another terminal."
                        )
                    )
                    time.sleep(min(dash.refresh_seconds, 5.0))
                    continue

                n_trades = len(trades)
                # ``trades["exchange"]`` may be typed as ``Series | DataFrame`` by
                # pyright; ``unique()`` reliably returns a numpy array we can len().
                n_exchanges = len(trades["exchange"].unique())
                n_symbols = len(trades["symbol"].unique())

                try:
                    results = analyze_all(trades, eff_analyzer)
                except Exception as exc:
                    log.warning("dashboard: analyze_all failed", error=str(exc))
                    results = []

                cycle_seconds = time.monotonic() - t_start
                table = render_table(
                    results,
                    config.report,
                    dash,
                    n_trades=n_trades,
                    n_exchanges=n_exchanges,
                    n_symbols=n_symbols,
                    cycle_seconds=cycle_seconds,
                )
                live.update(table)

                elapsed = time.monotonic() - t_start
                sleep_for = max(0.0, dash.refresh_seconds - elapsed)
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        console.print("[cyan]dashboard exited[/cyan]")
