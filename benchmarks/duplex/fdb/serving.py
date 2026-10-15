# SPDX-License-Identifier: Apache-2.0
"""Serving metrics: record one fixed v1.5 selection at several concurrency
levels against one model server with the request event recorder on, then
tabulate each level's load figures and server-side Unit timings."""

from __future__ import annotations

import argparse
import json

from benchmarks.duplex.fdb.common import Settings, log, log_tail
from benchmarks.duplex.fdb.generate import record_shards, v15_record_command
from benchmarks.duplex.fdb.selection import (
    SampleSelection,
    describe,
    ids_text,
    select_samples,
)
from benchmarks.duplex.fdb.servers import model_server
from benchmarks.duplex.run_artifacts import load_run
from benchmarks.duplex.serving_metrics import render_serving, serving_row
from benchmarks.eval.asr_profiling import start_request_profile, stop_request_profile

SERVING_DIR = "serving"


def parse_levels(text: str) -> list[int]:
    """Comma-separated concurrency levels such as 1,2,4,8, each at least 1 and
    given once."""
    parts = text.split(",")
    if not all(part.isdigit() and int(part) >= 1 for part in parts):
        raise argparse.ArgumentTypeError(
            f"expected positive integers separated by commas, such as 1,2,4,8; got '{text}'"
        )
    else:
        pass
    levels = [int(part) for part in parts]
    if len(set(levels)) != len(levels):
        raise argparse.ArgumentTypeError(f"each level must appear once, got '{text}'")
    else:
        return levels


def serving(settings: Settings, selection: SampleSelection, levels: list[int]) -> None:
    serving_dir = settings.run_root / SERVING_DIR
    if serving_dir.exists():
        raise SystemExit(
            f"ERROR: {serving_dir} already holds a serving sweep. "
            f"Delete it to redo the sweep, or use a new --run-name."
        )
    else:
        pass
    sample_ids = select_samples(settings.dataset, selection)
    if not sample_ids:
        raise SystemExit(
            "ERROR: no v1.5 pairs selected; give --v15-per-subset a nonzero count."
        )
    else:
        pass
    record = v15_record_command(settings)
    (serving_dir / "logs").mkdir(parents=True)
    (serving_dir / "sample-ids.txt").write_text(ids_text(sample_ids))
    log(f"== Selected {len(sample_ids)} v1.5 pairs: {describe(sample_ids)}")
    rows = []
    with model_server(
        settings,
        serving_dir / "logs" / "model-server.log",
        "--max_sessions",
        str(max(levels)),
    ) as server:
        for level in levels:
            level_dir = serving_dir / f"concurrency-{level}"
            event_dir = level_dir / "events"
            run_id = f"{settings.run_name}-concurrency-{level}"
            (level_dir / "logs").mkdir(parents=True)
            start_request_profile(settings.server_url, run_id, str(event_dir))
            log(
                f"== Recording {len(sample_ids)} v1.5 pairs "
                f"({2 * len(sample_ids)} sessions) at concurrency {level} "
                f"-> {level_dir}"
            )
            [recording] = record_shards(
                [*record, "--concurrency", str(level)], level_dir, sample_ids, 2, 1
            )
            stop_request_profile(settings.server_url, run_id)
            server.ensure_alive(f"at concurrency {level}")
            if not (recording / "run.json").is_file():
                record_log = level_dir / "logs" / f"record-{recording.name}.log"
                raise SystemExit(
                    f"ERROR: the recorder wrote no run.json at concurrency {level}; "
                    f"last lines of {record_log}:\n{log_tail(record_log)}"
                )
            else:
                pass
            if not any(event_dir.glob("events_*.jsonl")):
                raise SystemExit(
                    f"ERROR: the request event recorder wrote no events to "
                    f"{event_dir}; see {server.log_file}"
                )
            else:
                pass
            manifest, run, _ = load_run(recording)
            rows.append(serving_row(manifest, run, event_dir))
    table = render_serving(settings.run_name, rows)
    serving_json = settings.run_root / "serving.json"
    serving_json.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    serving_md = settings.run_root / "SERVING.md"
    serving_md.write_text(table, encoding="utf-8")
    log(table)
    log(f"Wrote {serving_json} and {serving_md}")
