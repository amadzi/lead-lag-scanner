"""Report generation: Markdown summary + machine-readable JSON config.

The Markdown report is sorted by tradeability score (descending) and
includes a naive bps-edge proxy, a net-of-cost edge, a tradeability
score, and a comma-separated list of sanity flags. A second table
surfaces per-exchange data-quality diagnostics (median transport latency
and the clock-skew offset the analyzer applied). The JSON file contains
the same data in a forward-compatible schema that downstream execution
bots can consume.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .analyzer import ExchangeDiagnostics, LeadLagResult
from .config import ReportConfig

MARKDOWN_HEADER = """\
# lead-lag-scanner report

Generated: {generated_at}

| Symbol | Leader | Follower | Lag (s) | Lag CI (s) | |Corr| | Corr@0 | Edge (bps) | Net (bps) | Score | Flags | Active A/B | n |
|--------|--------|----------|---------|------------|-------|--------|------------|-----------|-------|-------|------------|---|
"""

DIAGNOSTICS_HEADER = """\

## Per-exchange data quality

`transport_latency_p50` is the median of `local_recv − exchange_ts` and
captures both network RTT and the exchange's clock skew vs our local
clock. The analyzer subtracts the *cross-exchange median* of these
values from each timestamp, so `clock_offset_applied` is the residual
skew that was applied to align this exchange to the consensus timeline
(positive = pushed forward in time, negative = pulled earlier).

| Exchange | Trades | Transport latency p50 (ms) | Clock offset applied (ms) |
|----------|-------:|---------------------------:|--------------------------:|
"""


def _edge_bps(result: LeadLagResult, taker_bps: float) -> float:
    """Naive bps-edge proxy: ``|corr| × σ(follower) × 1e4 − 2 × taker_bps``.

    ``σ`` is the standard deviation of the follower's per-bar log-returns,
    so ``σ × 1e4`` is the per-bar move in basis points. Multiplying by the
    correlation gives the slice of that move we expect to capture.
    """

    edge = abs(result.best_correlation) * result.follower_return_std * 1e4
    return float(edge - 2.0 * taker_bps)


def _gross_edge_bps(result: LeadLagResult) -> float:
    """Edge before *any* costs: ``|corr| × σ(follower) × 1e4``.

    Useful as the numerator for tradeability and for showing the user how
    much theoretical pickup exists *before* fees, spread, or slippage are
    netted out. Always non-negative.
    """

    return float(abs(result.best_correlation) * result.follower_return_std * 1e4)


def _net_edge_bps(result: LeadLagResult, config: ReportConfig) -> float:
    """Edge after taker fee, spread, and slippage are netted out.

    All three cost components are *round-trip* totals (entry + exit)
    expressed in basis points, configurable via :class:`ReportConfig`.
    Default assumptions (4 bps spread, 2 bps slippage) are deliberately
    optimistic for tier-1 venues and pessimistic for microcaps; tune for
    your venue mix.
    """

    return float(
        _gross_edge_bps(result) - 2.0 * config.taker_bps - config.spread_bps - config.slippage_bps
    )


def _tradeability(result: LeadLagResult, config: ReportConfig) -> float:
    """Composite ranking metric in *bps × √obs* units.

    The score multiplies the net-of-cost edge by signal-quality factors:

    * ``correlation²`` — squared so noisy 0.4-corr pairs are penalised
      far more than the 0.1 difference would suggest;
    * ``√n_obs`` — observations only help as the square root, matching
      the standard error of a sample correlation;
    * ``leader_factor`` — 0 if the result has no directional leader (the
      lag is exactly zero), so undirected pairs sort to the bottom even
      when their correlation is high.

    Pairs with a negative net edge get a *negative* score and sort below
    pairs that at least cover their own costs. The score is zero (not
    negative) for undirected pairs because the "cost" of a no-trade is
    zero, not negative.
    """

    if result.leader == "none":
        return 0.0
    net = _net_edge_bps(result, config)
    corr_sq = float(result.best_correlation) ** 2
    obs_factor = math.sqrt(max(1, int(result.n_obs)))
    return float(net * corr_sq * obs_factor)


def _format_row(result: LeadLagResult, config: ReportConfig) -> str:
    edge = _edge_bps(result, config.taker_bps)
    net = _net_edge_bps(result, config)
    score = _tradeability(result, config)
    if result.leader == "none":
        leader_disp = follower_disp = "—"
    else:
        leader_disp, follower_disp = result.leader, result.follower
    ci_low = "—" if math.isnan(result.lag_ci_low_seconds) else f"{result.lag_ci_low_seconds:+.2f}"
    ci_high = (
        "—" if math.isnan(result.lag_ci_high_seconds) else f"{result.lag_ci_high_seconds:+.2f}"
    )
    active = f"{result.active_rate_a * 100:.0f}%/{result.active_rate_b * 100:.0f}%"
    flags = ",".join(result.flags) if result.flags else "—"
    return (
        f"| {result.symbol} "
        f"| {leader_disp} "
        f"| {follower_disp} "
        f"| {result.best_lag_seconds:+.2f} "
        f"| [{ci_low}, {ci_high}] "
        f"| {abs(result.best_correlation):.3f} "
        f"| {result.correlation_at_zero:.3f} "
        f"| {edge:+.1f} "
        f"| {net:+.1f} "
        f"| {score:+.0f} "
        f"| {flags} "
        f"| {active} "
        f"| {result.n_obs} |"
    )


def _format_diagnostic_row(d: ExchangeDiagnostics) -> str:
    return (
        f"| {d.exchange} "
        f"| {d.n_trades} "
        f"| {d.transport_latency_p50_ms:+.1f} "
        f"| {d.clock_offset_ms:+.1f} |"
    )


def write_reports(
    results: list[LeadLagResult],
    config: ReportConfig,
    *,
    diagnostics: list[ExchangeDiagnostics] | None = None,
) -> tuple[Path, Path]:
    """Write Markdown + JSON reports and return their paths.

    ``diagnostics`` is optional; when present it is emitted as a second
    Markdown table and as a ``"diagnostics"`` array in the JSON payload.
    """

    config.markdown_path.parent.mkdir(parents=True, exist_ok=True)
    config.json_path.parent.mkdir(parents=True, exist_ok=True)

    filtered = [r for r in results if abs(r.best_correlation) >= config.min_abs_correlation]
    # Sort by tradeability (best first); ties go to the better correlation so
    # the table is readable even when many pairs share a near-zero score.
    filtered.sort(
        key=lambda r: (-_tradeability(r, config), -abs(r.best_correlation)),
    )

    md_parts = [
        MARKDOWN_HEADER.format(generated_at=datetime.now(UTC).isoformat(timespec="seconds"))
    ]
    if not filtered:
        md_parts.append(
            "_No pairs met the correlation / observation thresholds._\n"
            "Try collecting more data or lowering ``report.min_abs_correlation``.\n"
        )
    else:
        for r in filtered:
            md_parts.append(_format_row(r, config))
        md_parts.append("")
        md_parts.append(
            "> Edge (bps) is the naive proxy: "
            "`|corr| × σ(follower) × 1e4 − 2 × taker_bps`.\n"
            "> Net (bps) subtracts spread and slippage on top of taker fees "
            "using `report.spread_bps` / `report.slippage_bps` (round-trip).\n"
            "> Score is the tradeability ranking metric "
            "`net_bps × corr² × √n_obs`; ≤ 0 means it does not cover its costs.\n"
            "> Flags surface statistical sanity issues — see `_compute_flags` "
            "in `analyzer.py` for definitions; `boundary_lag` and `wide_ci` "
            "are the loudest red flags.\n"
            "> `Active A/B` is the fraction of overlapping bars in which each "
            "side had a *real* trade (not forward-filled); pairs are dropped "
            "when either side is below `analyzer.min_active_rate`.\n"
            "> Treat any positive Score as *worth backtesting properly*, not "
            "as guaranteed PnL.\n"
        )

    if diagnostics:
        md_parts.append(DIAGNOSTICS_HEADER)
        for d in diagnostics:
            md_parts.append(_format_diagnostic_row(d))
        md_parts.append("")

    config.markdown_path.write_text("\n".join(md_parts), encoding="utf-8")

    json_payload: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "taker_bps_assumption": config.taker_bps,
        "spread_bps_assumption": config.spread_bps,
        "slippage_bps_assumption": config.slippage_bps,
        "min_abs_correlation": config.min_abs_correlation,
        "pairs": [
            {
                **asdict(r),
                "edge_bps": _edge_bps(r, config.taker_bps),
                "gross_edge_bps": _gross_edge_bps(r),
                "net_edge_bps": _net_edge_bps(r, config),
                "tradeability": _tradeability(r, config),
            }
            for r in filtered
        ],
    }
    if diagnostics:
        json_payload["diagnostics"] = [asdict(d) for d in diagnostics]
    config.json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    return config.markdown_path, config.json_path
