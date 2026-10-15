# SPDX-License-Identifier: Apache-2.0
"""Report serving timing of one duplex system across recorded concurrency levels."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import JsonValue

from benchmarks.duplex.serving import serving_report
from benchmarks.duplex.v15_audio import write_json

# note (jeffro): this is a small default startup buffer, and other values can be passed to trace the underrun curve.
DEFAULT_PLAYBACK_RESERVE_MS = 80.0


def milliseconds(value_s: JsonValue) -> str:
    return "-" if value_s is None else f"{value_s * 1000:.0f}"


def format_table(report: dict[str, JsonValue]) -> str:
    rows = [
        "| N | mean open | sessions failed | 503 | pacing violations | unit miss "
        "| unit lag p50 / p90 / p99 (ms) | first audio p50 / p90 (ms) "
        "| buffer needed p50 / p90 (ms) | underrun replies | meets targets |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for level in report["levels"]:
        unit_lag = level["unit_lag_s"]
        first_audio = level["first_audio_latency_s"]
        buffer_needed = level["startup_buffer_needed_s"]
        rows.append(
            f"| {level['concurrency']} | {level['mean_open_sessions']:.1f} "
            f"| {level['sessions_failed']}/{level['sessions']} "
            f"| {level['admission_denials']} | {level['input_pacing_violations']} "
            f"| {level['unit_miss_rate']:.1%} "
            f"| {milliseconds(unit_lag['p50'])} / {milliseconds(unit_lag['p90'])} / "
            f"{milliseconds(unit_lag['p99'])} "
            f"| {milliseconds(first_audio['p50'])} / {milliseconds(first_audio['p90'])} "
            f"| {milliseconds(buffer_needed['p50'])} / "
            f"{milliseconds(buffer_needed['p90'])} "
            f"| {level['replies_with_underrun']}/{level['replies']} "
            f"| {'yes' if not level['unmet_targets'] else '; '.join(level['unmet_targets'])} |"
        )
    capacity = report["max_concurrency_meeting_targets"]
    if capacity is None:
        verdict = "No recorded concurrency meets the targets."
    elif report["capacity_bounded"]:
        verdict = f"Highest concurrency meeting the targets: {capacity}."
    else:
        verdict = (
            f"Every recorded concurrency meets the targets; capacity is at least "
            f"{capacity}."
        )
    return "\n".join(rows) + f"\n\n{verdict}\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        action="append",
        required=True,
        help="Recorded run directory, one per concurrency level",
    )
    parser.add_argument("--output", type=Path, required=True, help="serving.json path")
    parser.add_argument(
        "--playback-reserve-ms",
        type=float,
        default=DEFAULT_PLAYBACK_RESERVE_MS,
        help="Simulated player startup buffer (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    if args.playback_reserve_ms < 0:
        parser.error("--playback-reserve-ms must be nonnegative")
    else:
        pass
    report = serving_report(args.run, args.playback_reserve_ms / 1000)
    write_json(args.output, report)
    print(format_table(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
else:
    pass
