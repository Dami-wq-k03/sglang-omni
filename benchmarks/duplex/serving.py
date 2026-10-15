# SPDX-License-Identifier: Apache-2.0
"""Client-observed serving timing of recorded native duplex sessions."""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path

from pydantic import JsonValue

from benchmarks.duplex.client import PACKET_MS, SAMPLE_RATE, SEND_RECEIPTS_FILE
from benchmarks.duplex.oracle_models import MediaTime, TraceRecord, WireEvent
from benchmarks.duplex.profiles import PROFILES, ProfileName

INPUT_BYTES_PER_MS = SAMPLE_RATE * 2 / 1000
# note (jeffro): the simulated player starts a reply 80 ms after its first audio
# arrives, so arrival jitter shorter than that is not counted as underrun.
PLAYBACK_RESERVE_S = 0.08


def score_session(variant_dir: Path, profile: ProfileName) -> dict[str, JsonValue]:
    """Unit lag, deadline misses and per-reply audio timing as the client saw them.

    A unit is ready when the client finished sending the packet that completes it,
    or, for the padded last unit, when it sent the input end.
    """
    unit_ms = PROFILES[profile].native_unit_ms
    output_sample_rate = PROFILES[profile].output_sample_rate
    packet_completed_s = {
        receipt["seq"]: receipt["completed_s"]
        for receipt in json.loads((variant_dir / SEND_RECEIPTS_FILE).read_text())[
            "appends"
        ]
    }
    input_ms = (variant_dir / "input.pcm").stat().st_size / INPUT_BYTES_PER_MS
    records = [
        TraceRecord.model_validate(json.loads(line))
        for line in (variant_dir / "continuous.jsonl").read_text().splitlines()
    ]
    input_end_s = next(
        (
            record.time_s
            for record in records
            if record.direction == "send"
            and record.event["type"] == "sglang.input_audio.end"
        ),
        None,
    )

    def unit_media_time(event: WireEvent) -> MediaTime:
        if event.sglang.media_time is None:
            raise ValueError(f"{variant_dir}: {event.type} has no media time")
        else:
            return event.sglang.media_time

    def ready_s(media_time: MediaTime) -> float:
        end_ms = media_time.t_start_ms + media_time.duration_ms
        packet_ready_s = packet_completed_s[math.ceil(end_ms / PACKET_MS) - 1]
        if media_time.duration_ms >= unit_ms:
            return packet_ready_s
        elif input_end_s is None:
            raise ValueError(f"{variant_dir}: a padded unit finished before input end")
        else:
            return max(packet_ready_s, input_end_s)

    unit_lags_s: list[float] = []
    reply_opening_unit: dict[str, MediaTime] = {}
    reply_first_audio_unit: dict[str, MediaTime] = {}
    reply_chunks: dict[str, list[tuple[float, float]]] = {}
    for record in (record for record in records if record.direction == "receive"):
        event = WireEvent.model_validate(record.event)
        if event.type == "sglang.unit.done":
            unit_lags_s.append(record.time_s - ready_s(unit_media_time(event)))
        elif event.type == "response.created":
            assert event.response is not None
            reply_opening_unit[event.response.id] = unit_media_time(event)
        elif event.type == "response.output_audio.delta":
            assert event.response_id is not None and event.delta is not None
            reply_first_audio_unit.setdefault(event.response_id, unit_media_time(event))
            reply_chunks.setdefault(event.response_id, []).append(
                (
                    record.time_s,
                    len(base64.b64decode(event.delta)) / 2 / output_sample_rate,
                )
            )
        else:
            pass

    replies: list[dict[str, JsonValue]] = []
    for response_id, chunks in reply_chunks.items():
        playback_s = chunks[0][0] + PLAYBACK_RESERVE_S
        underrun_s = 0.0
        for arrival_s, audio_s in chunks:
            underrun_s += max(0.0, arrival_s - playback_s)
            playback_s = max(playback_s, arrival_s) + audio_s
        # note (jeffro): the last chunk's own playback is not part of the generation
        # time, so the factor covers the audio that arrived before it.
        streamed_audio_s = sum(audio_s for _, audio_s in chunks[:-1])
        first_audio_unit = reply_first_audio_unit[response_id]
        replies.append(
            {
                "response_id": response_id,
                # note (jeffro): a reply can open with text one unit before its first
                # audio. That wait is one unit long by model design, whatever the load,
                # so latency starts at the unit carrying the first audio and the wait
                # is counted on its own.
                "first_audio_latency_s": chunks[0][0] - ready_s(first_audio_unit),
                "units_before_first_audio": round(
                    (
                        first_audio_unit.t_start_ms
                        - reply_opening_unit[response_id].t_start_ms
                    )
                    / unit_ms
                ),
                "audio_s": sum(audio_s for _, audio_s in chunks),
                "underrun_s": underrun_s,
                "real_time_factor": (
                    (chunks[-1][0] - chunks[0][0]) / streamed_audio_s
                    if len(chunks) > 1
                    else None
                ),
            }
        )

    units_expected = math.ceil(input_ms / unit_ms)
    return {
        "unit_ms": unit_ms,
        "units_expected": units_expected,
        "units_done": len(unit_lags_s),
        "units_missed": units_expected
        - len(unit_lags_s)
        + sum(lag_s > unit_ms / 1000 for lag_s in unit_lags_s),
        "unit_lag_s": unit_lags_s,
        "replies": replies,
    }
