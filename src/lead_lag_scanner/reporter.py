"""Report generation: Markdown summary + machine-readable JSON config.

The Markdown report is sorted by absolute correlation (descending) and
includes a naive bps-edge proxy. The JSON file contains the same data in a
forward-compatible schema that downstream execution bots can consume.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .analyzer import LeadLagResult
from .config import ReportConfig

MARKDOWN_HEADER = """\
# lead-lag-scanner report

Generated: {generated_at}

| Symbol | Leader | Follower | Lag (s) | Lag CI (s) | |Corr| | Corr@0 | Edge (bps) | n |
|--------|--------|----------|---------|------------|-------|--------|------------|---|
"""


def _edge_bps(result: LeadLagResult, taker_bps: float) -> float:
    """Naive bps-edge proxy: ``|corr| × σ(follower) × 1e4 − 2 × taker_bps``.

    ``σ`` is the standard deviation of the follower's per-bar log-returns,
    so ``σ × 1e4`` is the per-bar move in basis points. Multiplying by the
    correlation gives the slice of that move we expect to capture.
    """

    edge = abs(result.best_correlation) * result.follower_return_std * 1e4
    return float(edge - 2.0 * taker_bps)


def _format_row(result: LeadLagResult, taker_bps: float) -> str:
    edge = _edge_bps(result, taker_bps)
    if result.leader == "none":
        leader_disp = follower_disp = "—"
    else:
        leader_disp, follower_disp = result.leader, result.follower
    ci_low = (
        "—"
        if result.lag_ci_low_seconds != result.lag_ci_low_seconds
        else f"{result.lag_ci_low_seconds:+.2f}"
    )
    ci_high = (
        "—"
        if result.lag_ci_high_seconds != result.lag_ci_high_seconds
        else f"{result.lag_ci_high_seconds:+.2f}"
    )
    return (
        f"| {result.symbol} "
        f"| {leader_disp} "
        f"| {follower_disp} "
        f"| {result.best_lag_seconds:+.2f} "
        f"| [{ci_low}, {ci_high}] "
        f"| {abs(result.best_correlation):.3f} "
        f"| {result.correlation_at_zero:.3f} "
        f"| {edge:+.1f} "
        f"| {result.n_obs} |"
    )


def write_reports(
    results: list[LeadLagResult],
    config: ReportConfig,
) -> tuple[Path, Path]:
    """Write Markdown + JSON reports and return their paths."""

    config.markdown_path.parent.mkdir(parents=True, exist_ok=True)
    config.json_path.parent.mkdir(parents=True, exist_ok=True)

    filtered = [r for r in results if abs(r.best_correlation) >= config.min_abs_correlation]
    filtered.sort(key=lambda r: abs(r.best_correlation), reverse=True)

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
            md_parts.append(_format_row(r, config.taker_bps))
        md_parts.append("")
        md_parts.append(
            "> Edge (bps) is a naive proxy: `|corr| × σ(follower) × 1e4 − 2 × taker_bps`.\n"
            "> Treat any positive value as *worth backtesting properly*, not as guaranteed PnL.\n"
        )

    config.markdown_path.write_text("\n".join(md_parts), encoding="utf-8")

    json_payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "taker_bps_assumption": config.taker_bps,
        "min_abs_correlation": config.min_abs_correlation,
        "pairs": [
            {
                **asdict(r),
                "edge_bps": _edge_bps(r, config.taker_bps),
            }
            for r in filtered
        ],
    }
    config.json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    return config.markdown_path, config.json_path
