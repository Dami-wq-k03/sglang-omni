# SPDX-License-Identifier: Apache-2.0
"""The runbook's serving step: the serving-metrics row of one concurrency level
from synthetic recorder and event files, and the server, profiler and recorder
sequence of the sweep, without a GPU or a server."""

from __future__ import annotations

import argparse
import contextlib
import json
import types
from collections.abc import Iterator
from pathlib import Path

import pytest

from benchmarks.duplex.fdb import serving as serving_module
from benchmarks.duplex.fdb.__main__ import V15, build_parser, dataset_selection
from benchmarks.duplex.fdb.common import RECORD_PROFILE, Settings, load_settings
from benchmarks.duplex.fdb.selection import SampleSelection, ids_text
from benchmarks.duplex.profiles import PROFILES
from benchmarks.duplex.serving_metrics import render_serving, serving_row
from benchmarks.duplex.v15_dataset import SUBSETS as V15_SUBSETS
from benchmarks.duplex.v15_runner import RUN_KIND

NOMINAL_UNIT_MS = PROFILES[RECORD_PROFILE].native_unit_ms
LEVELS = [1, 2, 4, 8]


def unit_events(
    session_id: str,
    input_seq: int,
    *,
    media_duration_ms: int,
    first_output_ms: int,
    thinker_service_ms: int,
    finished_ms: int | None,
) -> list[dict]:
    """The events of one Unit that becomes ready at input_seq nominal Units into
    its session; the other offsets are milliseconds after ready."""
    request_id = f"{session_id}-{input_seq}"
    ready_ns = input_seq * NOMINAL_UNIT_MS * 1_000_000
    identity = {
        "session_id": session_id,
        "session_open_index": 0,
        "input_seq": input_seq,
    }
    media = {
        **identity,
        "input_modality": "audio",
        "input_t_start_ms": input_seq * NOMINAL_UNIT_MS,
        "input_duration_ms": media_duration_ms,
    }

    def event(name: str, after_ms: int, stage: str, metadata: dict) -> dict:
        return {
            "request_id": request_id,
            "stage": stage,
            "event_name": name,
            "timestamp_ns": ready_ns + after_ms * 1_000_000,
            "run_id": "fixture",
            "pid": 1,
            "metadata": metadata,
        }

    events = [
        event("session_unit_ready", 0, "coordinator", media),
        event("session_unit_admitted", 1, "coordinator", media),
        event("stage_dispatch", 2, "thinker", {}),
        event("session_stage_started", 3, "thinker", identity),
        event("stage_complete", 3 + thinker_service_ms, "thinker", {}),
        event(
            "session_output_emitted",
            first_output_ms,
            "coordinator",
            {**identity, "kind": "audio", "output_seq": 0},
        ),
    ]
    if finished_ms is not None:
        events.append(event("session_unit_finished", finished_ms, "coordinator", media))
    else:
        pass
    return events


def write_events(event_dir: Path) -> None:
    """Four Units: a 1000 ms Unit within its deadline, one past it, the short
    last Unit of that session, and a Unit of a second session that never
    finished."""
    events = [
        *unit_events(
            "a",
            0,
            media_duration_ms=1000,
            first_output_ms=100,
            thinker_service_ms=40,
            finished_ms=300,
        ),
        *unit_events(
            "a",
            1,
            media_duration_ms=1000,
            first_output_ms=120,
            thinker_service_ms=60,
            finished_ms=1200,
        ),
        *unit_events(
            "a",
            2,
            media_duration_ms=200,
            first_output_ms=160,
            thinker_service_ms=80,
            finished_ms=400,
        ),
        *unit_events(
            "b",
            0,
            media_duration_ms=1000,
            first_output_ms=200,
            thinker_service_ms=100,
            finished_ms=None,
        ),
    ]
    event_dir.mkdir(parents=True)
    (event_dir / "events_coordinator_1.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )


def recorder_run(concurrency: int) -> dict:
    return {
        "status": "complete",
        "summary": {"variants_selected": 4},
        "load": {
            "concurrency": concurrency,
            "recording_wall_s": 30.0,
            "initial_delay_s": {"count": 4, "p50": 0.5, "p90": 0.9, "max": 1.25},
            "max_input_pacing_deviation_s": 0.002,
            "input_pacing_violations": 0,
            "variants_with_admission_denials": 1,
            "variants_passed": 3,
        },
        "samples": [],
    }


MANIFEST = {"kind": RUN_KIND, "profile": RECORD_PROFILE}


def test_row_counts_deadline_misses_against_the_nominal_unit(tmp_path: Path) -> None:
    event_dir = tmp_path / "events"
    write_events(event_dir)

    row = serving_row(MANIFEST, recorder_run(2), event_dir)

    assert row["nominal_unit_ms"] == NOMINAL_UNIT_MS == 1000
    assert (row["units"], row["finished_units"]) == (4, 3)
    # note (luojiaxuan): the last Unit of session a finishes 400 ms after it is
    # ready, past its own 200 ms of audio but within the nominal Unit, so only
    # the 1200 ms Unit misses; the unfinished Unit is not judged.
    assert row["deadline_misses"] == 1
    assert row["ready_to_finished_ms"] == {
        "count": 3,
        "p50": 400.0,
        "p95": pytest.approx(1120.0),
        "max": 1200.0,
    }
    assert row["ready_to_first_output_ms"] == {
        "count": 4,
        "p50": 140.0,
        "p95": pytest.approx(194.0),
        "max": 200.0,
    }
    assert row["thinker_service_ms_p95"] == pytest.approx(97.0)
    assert row["concurrency"] == 2
    assert (row["variants_passed"], row["variants_selected"]) == (3, 4)
    assert row["input_pacing_violations"] == 0
    assert row["variants_with_admission_denials"] == 1
    assert row["initial_delay_s"] == {"count": 4, "p50": 0.5, "p90": 0.9, "max": 1.25}
    assert render_serving("sweep", [row]).splitlines()[-1] == (
        "| 2 | 3/4 | 0 | 1 | 0.500 / 0.900 / 1.250 | 4 / 3 | 140 / 194 / 200 "
        "| 400 / 1120 / 1200 | 97 | 1 |"
    )


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    for name in ("CUDA_VISIBLE_DEVICES", "GPU", "SERVER_PORT", "JUDGE_PORT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FDB_WORK", str(tmp_path / "fdb"))
    settings = load_settings("sweep")
    for subset in V15_SUBSETS:
        for name in ("1", "2", "3", "4", "5"):
            (settings.dataset / subset / name).mkdir(parents=True)
    settings.dataset_revision_file.write_text("sha256:v15\n")
    return settings


def stub_sweep(
    monkeypatch: pytest.MonkeyPatch, server_records_events: bool = True
) -> tuple[list[tuple], list[tuple]]:
    """Replace the server, the profiler control and the recorder. Each fake
    recording writes a recorder run and, unless server_records_events is
    False, the events the server would record."""
    servers: list[tuple] = []
    calls: list[tuple] = []
    server = types.SimpleNamespace(
        log_file=Path("model-server.log"),
        ensure_alive=lambda when: calls.append(("alive", when)),
    )

    @contextlib.contextmanager
    def model_server(*arguments: object) -> Iterator[types.SimpleNamespace]:
        servers.append(arguments)
        yield server

    def record_shards(
        base_command: list[str],
        root: Path,
        sample_ids: list[str],
        sessions_per_sample: int,
        num_shards: int,
    ) -> list[Path]:
        calls.append(
            ("record", base_command, root, sample_ids, sessions_per_sample, num_shards)
        )
        recording = root / "recording" / "shard-0"
        recording.mkdir(parents=True)
        level = int(base_command[base_command.index("--concurrency") + 1])
        (recording / "manifest.json").write_text(json.dumps(MANIFEST))
        (recording / "run.json").write_text(json.dumps(recorder_run(level)))
        if server_records_events:
            write_events(root / "events")
        else:
            pass
        return [recording]

    monkeypatch.setattr(serving_module, "model_server", model_server)
    monkeypatch.setattr(serving_module, "record_shards", record_shards)
    monkeypatch.setattr(
        serving_module,
        "start_request_profile",
        lambda url, run_id, event_dir: calls.append(("start", url, run_id, event_dir)),
    )
    monkeypatch.setattr(
        serving_module,
        "stop_request_profile",
        lambda url, run_id: calls.append(("stop", url, run_id)),
    )
    return servers, calls


def test_sweep_serves_once_and_records_each_level_under_its_own_profiler_run(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    servers, calls = stub_sweep(monkeypatch)
    sample_ids = [f"{subset}/{name}" for subset in V15_SUBSETS for name in "1234"]
    serving_dir = settings.run_root / "serving"

    serving_module.serving(settings, SampleSelection(per_subset=4), LEVELS)

    assert servers == [
        (
            settings,
            serving_dir / "logs" / "model-server.log",
            "--max_sessions",
            "8",
        )
    ]
    record = serving_module.v15_record_command(settings)
    expected = []
    for level in LEVELS:
        level_dir = serving_dir / f"concurrency-{level}"
        run_id = f"sweep-concurrency-{level}"
        expected += [
            ("start", settings.server_url, run_id, str(level_dir / "events")),
            (
                "record",
                [*record, "--concurrency", str(level)],
                level_dir,
                sample_ids,
                2,
                1,
            ),
            ("stop", settings.server_url, run_id),
            ("alive", f"at concurrency {level}"),
        ]
    assert calls == expected
    assert settings.server_url == f"http://127.0.0.1:{settings.server_port}"
    assert (serving_dir / "sample-ids.txt").read_text() == ids_text(sample_ids)
    rows = json.loads((settings.run_root / "serving.json").read_text())
    assert [row["concurrency"] for row in rows] == LEVELS
    assert [row["deadline_misses"] for row in rows] == [1] * len(LEVELS)
    table = (settings.run_root / "SERVING.md").read_text()
    assert [line.split(" | ")[0] for line in table.splitlines()[-4:]] == [
        f"| {level}" for level in LEVELS
    ]

    with pytest.raises(SystemExit, match="already holds a serving sweep"):
        serving_module.serving(settings, SampleSelection(per_subset=4), LEVELS)
    assert len(servers) == 1


def test_sweep_stops_when_the_event_recorder_wrote_nothing(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, calls = stub_sweep(monkeypatch, server_records_events=False)
    with pytest.raises(SystemExit, match="request event recorder wrote no events"):
        serving_module.serving(settings, SampleSelection(per_subset=1), [1, 2])
    assert [call[0] for call in calls] == ["start", "record", "stop", "alive"]


def test_serving_options_take_levels_and_a_v15_only_selection() -> None:
    parser = build_parser()
    args = parser.parse_args(["serving", "--concurrency", "1,2,4,8"])
    assert args.concurrency == LEVELS
    assert dataset_selection(parser, args, V15) == SampleSelection(per_subset=4)
    args = parser.parse_args(
        [
            "serving",
            "--concurrency",
            "2",
            "--v15-subset-count",
            "user_interruption=8",
        ]
    )
    assert dataset_selection(parser, args, V15) == SampleSelection(
        per_subset=4, subset_counts={"user_interruption": 8}
    )
    for arguments in (
        ["serving"],
        ["serving", "--concurrency", "2", "--v10-per-subset", "1"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(arguments)
    for text in ("0,2", "1,,2", "two", "1,2,1"):
        with pytest.raises(argparse.ArgumentTypeError):
            serving_module.parse_levels(text)
