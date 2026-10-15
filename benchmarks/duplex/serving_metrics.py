# SPDX-License-Identifier: Apache-2.0
"""Serving metrics of one v1.5 recording at one concurrency level: the
recorder's client-side load figures joined with the server-side Unit timings
of the request event recorder."""

from __future__ import annotations

from pathlib import Path

from pydantic import JsonValue

from benchmarks.duplex.profiles import PROFILES
from sglang_omni.profiler.duplex_metrics import build_duplex_session_metrics

THINKER_STAGE = "thinker"


def serving_row(
    manifest: dict[str, JsonValue], run: dict[str, JsonValue], event_dir: Path
) -> dict[str, JsonValue]:
    """One table row from a recorder run's manifest.json and run.json and the
    event directory the request event recorder wrote during that run.

    A deadline miss is a finished Unit whose ready-to-finished time exceeds the
    profile's nominal Unit length. The next Unit is ready one nominal Unit
    later, whatever the audio length of this one, so the nominal length is the
    budget; the last Unit of a session carries only the remaining audio, and
    its own media duration would count false misses."""
    load = run["load"]
    metrics = build_duplex_session_metrics(event_dir)
    summary = metrics["summary"]
    units = metrics["units"]
    nominal_unit_ms = PROFILES[manifest["profile"]].native_unit_ms
    return {
        "concurrency": load["concurrency"],
        "variants_selected": run["summary"]["variants_selected"],
        "variants_passed": load["variants_passed"],
        "input_pacing_violations": load["input_pacing_violations"],
        "variants_with_admission_denials": load["variants_with_admission_denials"],
        "initial_delay_s": load["initial_delay_s"],
        "units": len(units),
        "finished_units": sum(
            unit["finished_timestamp_ns"] is not None for unit in units
        ),
        "ready_to_first_output_ms": summary["ready_to_first_output_ms"],
        "ready_to_finished_ms": summary["ready_to_finished_ms"],
        "thinker_service_ms_p95": summary["stages"][THINKER_STAGE]["service_ms"]["p95"],
        "nominal_unit_ms": nominal_unit_ms,
        "deadline_misses": sum(
            unit["ready_to_finished_ms"] is not None
            and unit["ready_to_finished_ms"] > nominal_unit_ms
            for unit in units
        ),
    }


def format_value(value: float | None, digits: int) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def format_spread(
    distribution: dict[str, JsonValue], keys: tuple[str, ...], digits: int
) -> str:
    return " / ".join(format_value(distribution[key], digits) for key in keys)


def render_serving(run_name: str, rows: list[dict[str, JsonValue]]) -> str:
    """SERVING.md: one row per concurrency level, in the order recorded."""
    nominal_unit_ms = rows[0]["nominal_unit_ms"]
    lines = [
        f"# FDB serving metrics: {run_name}",
        "",
        "Every level records the same v1.5 pairs against one model server. "
        "Initial delay, pacing and admission come from the recorder; Unit "
        "timings come from the server's request event recorder. A deadline miss "
        "is a finished Unit whose ready-to-finished time exceeds the nominal "
        f"{nominal_unit_ms} ms Unit. Latencies under load do not compare with "
        "behavior runs at concurrency 1.",
        "",
        "| Concurrency | Passed | Pacing violations | Variants with denials "
        "| Initial delay p50 / p90 / max (s) | Units / finished "
        "| Ready to first output p50 / p95 / max (ms) "
        "| Ready to finished p50 / p95 / max (ms) | Thinker service p95 (ms) "
        "| Deadline misses |",
        "|" + "---:|" * 10,
    ]
    for row in rows:
        cells = [
            str(row["concurrency"]),
            f"{row['variants_passed']}/{row['variants_selected']}",
            str(row["input_pacing_violations"]),
            str(row["variants_with_admission_denials"]),
            format_spread(row["initial_delay_s"], ("p50", "p90", "max"), 3),
            f"{row['units']} / {row['finished_units']}",
            format_spread(row["ready_to_first_output_ms"], ("p50", "p95", "max"), 0),
            format_spread(row["ready_to_finished_ms"], ("p50", "p95", "max"), 0),
            format_value(row["thinker_service_ms_p95"], 0),
            str(row["deadline_misses"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
