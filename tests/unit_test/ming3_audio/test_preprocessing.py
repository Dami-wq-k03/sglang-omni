# SPDX-License-Identifier: Apache-2.0
"""Audio decoding and resampling against analytic signals."""

import base64
import io
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import soundfile
import torch
from whisper.audio import SAMPLE_RATE as WHISPER_SAMPLE_RATE

from sglang_omni.models.ming3_audio.preprocessing import AudioMediaIO

STEREO_FRAME = np.array([[0.5, -0.25]], dtype=np.float32)
STEREO_FRAME_COUNT = 1600
MEAN_AMPLITUDE = 0.125
TONE_FREQUENCY_HZ = 440
TONE_AMPLITUDE = 0.25
SOURCE_SAMPLE_RATE = 48000
TRANSIENT_EDGE_SAMPLES = 32
RESAMPLE_ABSOLUTE_TOLERANCE = 2e-4


def stereo_channel_wav_bytes(subtype: Literal["PCM_16", "FLOAT"]) -> bytes:
    waveform = np.tile(STEREO_FRAME, (STEREO_FRAME_COUNT, 1))
    encoded = io.BytesIO()
    soundfile.write(
        encoded, waveform, WHISPER_SAMPLE_RATE, format="WAV", subtype=subtype
    )
    return encoded.getvalue()


@pytest.mark.parametrize("subtype", ["PCM_16", "FLOAT"])
def test_wav_bytes_downmix_stereo_channels_to_mean(
    subtype: Literal["PCM_16", "FLOAT"],
) -> None:
    actual = AudioMediaIO().load_bytes(stereo_channel_wav_bytes(subtype))
    torch.testing.assert_close(
        actual, torch.full((STEREO_FRAME_COUNT,), MEAN_AMPLITUDE), rtol=0, atol=0
    )


def test_base64_and_file_loaders_match_wav_bytes(tmp_path: Path) -> None:
    wav_bytes = stereo_channel_wav_bytes("FLOAT")
    media_io = AudioMediaIO()
    expected = media_io.load_bytes(wav_bytes)
    decoded_base64 = media_io.load_base64(
        "audio/wav", base64.b64encode(wav_bytes).decode()
    )
    wav_path = tmp_path / "reference.wav"
    wav_path.write_bytes(wav_bytes)
    decoded_file = media_io.load_file(wav_path)
    torch.testing.assert_close(decoded_base64, expected, rtol=0, atol=0)
    torch.testing.assert_close(decoded_file, expected, rtol=0, atol=0)


def test_resampling_preserves_tone_frequency_and_level() -> None:
    source_time = np.arange(SOURCE_SAMPLE_RATE) / SOURCE_SAMPLE_RATE
    waveform = TONE_AMPLITUDE * np.sin(2 * np.pi * TONE_FREQUENCY_HZ * source_time)
    encoded = io.BytesIO()
    soundfile.write(
        encoded, waveform, SOURCE_SAMPLE_RATE, format="WAV", subtype="FLOAT"
    )
    actual = AudioMediaIO().load_bytes(encoded.getvalue())
    target_time = torch.arange(WHISPER_SAMPLE_RATE) / WHISPER_SAMPLE_RATE
    expected = TONE_AMPLITUDE * torch.sin(
        2 * torch.pi * TONE_FREQUENCY_HZ * target_time
    )
    assert actual.shape == (WHISPER_SAMPLE_RATE,)
    interior = slice(TRANSIENT_EDGE_SAMPLES, -TRANSIENT_EDGE_SAMPLES)
    torch.testing.assert_close(
        actual[interior],
        expected[interior],
        rtol=0,
        atol=RESAMPLE_ABSOLUTE_TOLERANCE,
    )
