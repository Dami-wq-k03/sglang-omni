import base64
import json
import math
from pathlib import Path

import pytest
from pydantic import JsonValue

from benchmarks.duplex.client import SEND_RECEIPTS_FILE
from benchmarks.duplex.serving import score_session

# MiniCPM-o: 1000 ms units, 24 kHz output.
PROFILE = "minicpmo-native-pr2377"
PLAYBACK_RESERVE_S = 0.3
START_S = 100.0
INPUT_S = 2.5
OUTPUT_SECOND = base64.b64encode(b"\x00\x00" * 24000).decode()


def record_session(
    variant_dir: Path,
    unit_done_s: list[float],
    *,
    input_s: float = INPUT_S,
    reply_unit: int = 1,
    reply_audio_unit: int = 1,
    reply_audio_s: list[float] | None = None,
) -> Path:
    """Write a paced 80 ms session whose unit k completes at unit_done_s[k]."""
    variant_dir.mkdir()
    packets = math.ceil(input_s * 1000 / 80)
    (variant_dir / "input.pcm").write_bytes(b"\x00\x00" * round(input_s * 16000))
    (variant_dir / SEND_RECEIPTS_FILE).write_text(
        json.dumps(
            {
                "appends": [
                    {
                        "event_id": f"input{seq}",
                        "seq": seq,
                        "start_s": START_S + seq * 0.08,
                        "completed_s": START_S + seq * 0.08,
                    }
                    for seq in range(packets)
                ]
            }
        )
    )
    records: list[dict[str, JsonValue]] = []

    def add(time_s: float, direction: str, event_type: str, **event: JsonValue) -> None:
        records.append(
            {
                "direction": direction,
                "time_s": time_s,
                "event": {
                    "type": event_type,
                    "event_id": f"event_{len(records)}",
                    **event,
                },
            }
        )

    def media(unit: int) -> dict[str, JsonValue]:
        return {
            "unit_id": f"unit_{unit}",
            "media_time": {
                "t_start_ms": unit * 1000.0,
                "duration_ms": min(1000.0, input_s * 1000 - unit * 1000),
            },
        }

    add(START_S + input_s, "send", "sglang.input_audio.end")
    for unit, done_s in enumerate(unit_done_s):
        add(
            done_s,
            "receive",
            "sglang.unit.done",
            unit_id=f"unit_{unit}",
            sglang=media(unit),
        )
    if reply_audio_s is not None:
        add(
            reply_audio_s[0],
            "receive",
            "response.created",
            response={"id": "response0", "status": "in_progress"},
            sglang=media(reply_unit),
        )
        for arrival_s in reply_audio_s:
            add(
                arrival_s,
                "receive",
                "response.output_audio.delta",
                response_id="response0",
                delta=OUTPUT_SECOND,
                sglang=media(reply_audio_unit),
            )
    else:
        pass
    records.sort(key=lambda record: record["time_s"])
    (variant_dir / "continuous.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    return variant_dir


def test_unit_lag_counts_a_late_unit_and_waits_for_input_end(tmp_path: Path) -> None:
    # Unit 0 completes with packet 12 (100.96), unit 1 with packet 24 (101.92); the
    # padded 500 ms unit 2 waits for the input end at 102.5, not packet 31 (102.48).
    session = score_session(
        record_session(tmp_path / "session", [101.16, 103.0, 103.1]),
        PROFILE,
        PLAYBACK_RESERVE_S,
    )

    assert session["units_expected"] == 3
    assert session["units_done"] == 3
    assert session["unit_lag_s"] == pytest.approx([0.2, 1.08, 0.6])
    assert session["units_missed"] == 1


def test_unit_without_receipt_is_missed(tmp_path: Path) -> None:
    session = score_session(
        record_session(tmp_path / "session", [101.16, 102.12]),
        PROFILE,
        PLAYBACK_RESERVE_S,
    )

    assert session["units_done"] == 2
    assert session["units_missed"] == 1


@pytest.mark.parametrize(
    ("playback_reserve_s", "underrun_s"), [(0.0, 0.2), (0.08, 0.12), (0.3, 0.0)]
)
def test_reply_latency_underrun_and_real_time_factor(
    tmp_path: Path, playback_reserve_s: float, underrun_s: float
) -> None:
    session = score_session(
        record_session(
            tmp_path / "session",
            [101.16, 103.0, 103.1],
            reply_audio_s=[103.0, 104.2],
        ),
        PROFILE,
        playback_reserve_s,
    )

    (reply,) = session["replies"]
    assert reply["first_audio_latency_s"] == pytest.approx(1.08)
    assert reply["units_before_first_audio"] == 0
    assert reply["audio_s"] == pytest.approx(2.0)
    assert reply["underrun_s"] == pytest.approx(underrun_s)
    assert reply["startup_buffer_needed_s"] == pytest.approx(0.2)
    assert reply["real_time_factor"] == pytest.approx(1.2)


def test_reply_latency_starts_at_the_unit_with_the_first_audio(tmp_path: Path) -> None:
    session = score_session(
        record_session(
            tmp_path / "session",
            [101.16, 102.12, 102.7],
            reply_audio_unit=2,
            reply_audio_s=[102.7],
        ),
        PROFILE,
        PLAYBACK_RESERVE_S,
    )

    (reply,) = session["replies"]
    assert reply["first_audio_latency_s"] == pytest.approx(0.2)
    assert reply["units_before_first_audio"] == 1
    assert reply["real_time_factor"] is None
