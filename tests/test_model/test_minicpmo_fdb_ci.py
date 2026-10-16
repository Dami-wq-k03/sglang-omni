# SPDX-License-Identifier: Apache-2.0
"""Full-Duplex-Bench smoke for MiniCPM-o in the Omni CI lane.

The runbook records two v1.5 pairs per category and five v1.0 samples per
subset against two MiniCPM-o servers, one per GPU with one session each, then
transcribes and scores them without a judge: `asr` already writes the v1.5
timing and the v1.0 unified scores. The subset is sized to finish in about ten
minutes on two H100s. The test checks that every session kept the realtime protocol
and that every selected sample was scored. It sets no quality thresholds; a
handful of sampled generations cannot calibrate them.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

from huggingface_hub import snapshot_download

from benchmarks.duplex.fdb.common import DEFAULT_MODEL_REVISION, MODEL_ID
from benchmarks.duplex.v10_dataset import SUBSET_TASKS
from benchmarks.duplex.v15_dataset import SUBSETS as V15_CATEGORIES

RUN_NAME = "ci"
V15_PER_SUBSET = 2
V10_PER_SUBSET = 5
GPUS = "0,1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def runbook(env: dict[str, str], *arguments: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "benchmarks.duplex.fdb", *arguments],
        env=env,
        check=True,
    )


def variant_statuses(recording: Path) -> Counter:
    return Counter(
        variant["status"]
        for run in recording.glob("shard-*/run.json")
        for sample in read_json(run)["samples"]
        for variant in sample["variants"].values()
    )


def test_minicpmo_full_duplex_bench_smoke() -> None:
    fdb_work = Path(os.environ["OMNI_CI_HOME"]) / "fdb"
    env = {
        **os.environ,
        "FDB_WORK": str(fdb_work),
        "MODEL_PATH": snapshot_download(MODEL_ID, revision=DEFAULT_MODEL_REVISION),
        # note (luojiaxuan): with JUDGE=gpt, setup skips the 27B judge model; the
        # smoke never runs the judge step, so no API key is read.
        "JUDGE": "gpt",
    }
    run_root = fdb_work / "runs" / RUN_NAME
    # note (luojiaxuan): generate refuses a repeat that already holds a recording,
    # so a retried attempt starts from an empty run.
    if run_root.exists():
        shutil.rmtree(run_root)
    else:
        pass
    runbook(env, "setup")
    runbook(
        env,
        "generate",
        "--run-name",
        RUN_NAME,
        "--v15-per-subset",
        str(V15_PER_SUBSET),
        "--v10-per-subset",
        str(V10_PER_SUBSET),
        "--gpus",
        GPUS,
        "--num-shards",
        str(len(GPUS.split(","))),
    )
    runbook(env, "asr", "--run-name", RUN_NAME)
    runbook(env, "aggregate", "--run-name", RUN_NAME)

    repeat = run_root / "repeat-1"
    pairs = V15_PER_SUBSET * len(V15_CATEGORIES)
    v10_samples = V10_PER_SUBSET * len(SUBSET_TASKS)
    assert variant_statuses(repeat / "recording") == {"pass": 2 * pairs}
    assert variant_statuses(repeat / "v10" / "recording") == {"pass": v10_samples}
    manifest = read_json(repeat / "reference-audio" / "reference-manifest.json")
    assert manifest["counts"]["eligible_pairs"] == pairs

    samples = repeat / "scores" / "engines" / "minicpmo" / "samples"
    assert Counter(
        read_json(receipt)["status"]
        for receipt in samples.glob("*/*/receipts/timing-overlap.json")
    ) == {"ok": pairs}

    unified = read_json(repeat / "v10" / "reference" / "unified.json")
    # note (luojiaxuan): the unified scorer leaves backchannel to the official
    # evaluator, which needs the judge step.
    assert set(unified["subsets"]) == {
        subset for subset, task in SUBSET_TASKS.items() if task != "backchannel"
    }
    for subset, entry in unified["subsets"].items():
        assert entry["counts"]["selected"] == V10_PER_SUBSET, subset
        assert entry["counts"]["eligible"] == V10_PER_SUBSET, subset
    assert (run_root / "RESULTS.md").is_file()
