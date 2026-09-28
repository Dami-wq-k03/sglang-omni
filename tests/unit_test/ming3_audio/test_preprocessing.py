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

from sglang_omni.models.ming3_audio.preprocessing import AudioMediaIO


@pytest.mark.parametrize("subtype", ["PCM_16", "FLOAT"])
@pytest.mark.parametrize("source", ["bytes", "base64", "file"])
def test_audio_sources_preserve_stereo_amplitude(
    tmp_path: Path,
    subtype: Literal["PCM_16", "FLOAT"],
    source: Literal["bytes", "base64", "file"],
) -> None:
    waveform = np.tile(np.array([[0.5, -0.25]], dtype=np.float32), (1600, 1))
    encoded = io.BytesIO()
    soundfile.write(encoded, waveform, 16000, format="WAV", subtype=subtype)
    media_io = AudioMediaIO()
    if source == "bytes":
        actual = media_io.load_bytes(encoded.getvalue())
    elif source == "base64":
        actual = media_io.load_base64(
            "audio/wav", base64.b64encode(encoded.getvalue()).decode()
        )
    else:
        path = tmp_path / "reference.wav"
        path.write_bytes(encoded.getvalue())
        actual = media_io.load_file(path)
    torch.testing.assert_close(actual, torch.full((1600,), 0.125), rtol=0, atol=0)


def test_resampling_preserves_tone_frequency_and_level() -> None:
    frequency = 440
    waveform = 0.25 * np.sin(2 * np.pi * frequency * np.arange(48000) / 48000)
    encoded = io.BytesIO()
    soundfile.write(encoded, waveform, 48000, format="WAV", subtype="FLOAT")
    actual = AudioMediaIO().load_bytes(encoded.getvalue())
    expected = 0.25 * torch.sin(2 * torch.pi * frequency * torch.arange(16000) / 16000)
    assert actual.shape == (16000,)
    torch.testing.assert_close(actual[32:-32], expected[32:-32], rtol=0, atol=0.0002)
