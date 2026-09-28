# SPDX-License-Identifier: Apache-2.0
"""Chat prompts and audio features for Ming 3 audio."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import numpy.typing as npt
import torch
import whisper
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from sglang_omni.models.ming3_audio.audio_encoder import AudioEncoderConfig
from sglang_omni.preprocessing.audio import AudioMediaIO, ensure_audio_list_async
from sglang_omni.proto.request import StagePayload

AUDIO_PLACEHOLDER = "<|audio_pad|>"
ASR_PROMPT = "Please transcribe the speech into text. Only output the transcription."


class ContentPart(BaseModel):
    type: Literal["text", "audio"]
    text: str = ""
    audio: str | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart]


class AudioChatInputs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[ChatMessage] = Field(default_factory=list)
    audio: list[str | npt.NDArray[np.float32]] = Field(default_factory=list)
    audio_bytes: bytes | None = None


class AudioPipelineState(TypedDict):
    input_ids: torch.Tensor
    audio_positions: torch.Tensor
    features: list[torch.Tensor]
    audio_embeddings: torch.Tensor


class Ming3AudioPreprocessor:
    def __init__(self, model_path: str) -> None:
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=False
        )
        configuration = json.loads((Path(model_path) / "config.json").read_text())
        self.audio_config = AudioEncoderConfig.model_validate(
            configuration["audio_config"]
        )
        self.hidden_size: int = configuration["text_config"]["hidden_size"]
        self.audio_token_id: int = configuration["audio_token_id"]

    async def __call__(self, payload: StagePayload) -> StagePayload:
        inputs = AudioChatInputs.model_validate(payload.request.inputs)
        if inputs.audio_bytes is not None:
            waveform, _ = AudioMediaIO(target_sr=16000).load_bytes(inputs.audio_bytes)
            waveforms = [waveform]
            messages = [
                ChatMessage(
                    role="user",
                    content=[
                        ContentPart(type="audio"),
                        ContentPart(type="text", text=ASR_PROMPT),
                    ],
                )
            ]
        else:
            messages = inputs.messages
            sources = list(inputs.audio)
            for message in messages:
                if isinstance(message.content, list):
                    sources.extend(
                        part.audio
                        for part in message.content
                        if part.type == "audio" and part.audio is not None
                    )
                else:
                    pass
            waveforms = await ensure_audio_list_async(sources, target_sr=16000)

        features = [
            whisper.log_mel_spectrogram(
                torch.as_tensor(waveform), n_mels=self.audio_config.n_mels
            ).T.contiguous()
            for waveform in waveforms
        ]
        prompt = self.tokenizer.apply_chat_template(
            [message.model_dump(exclude_none=True) for message in messages],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if features and AUDIO_PLACEHOLDER not in prompt:
            # note (Codex): the HTTP adapter separates audio from message content.
            content = messages[-1].content
            if isinstance(content, str):
                content = [ContentPart(type="text", text=content)]
            else:
                pass
            messages[-1].content = content + [
                ContentPart(type="audio") for _ in features
            ]
            prompt = self.tokenizer.apply_chat_template(
                [message.model_dump(exclude_none=True) for message in messages],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            pass
        sections = prompt.split(AUDIO_PLACEHOLDER)
        if len(sections) != len(features) + 1:
            raise ValueError("Audio placeholders must match the number of audio inputs")
        else:
            pass
        expanded = sections[0]
        for mel, section in zip(features, sections[1:], strict=True):
            expanded += (
                AUDIO_PLACEHOLDER * self.audio_config.output_length(mel.shape[0])
                + section
            )
        input_ids = torch.tensor(self.tokenizer.encode(expanded), dtype=torch.long)
        payload.data = AudioPipelineState(
            input_ids=input_ids,
            audio_positions=torch.where(input_ids == self.audio_token_id)[0],
            features=features,
            audio_embeddings=torch.empty((0, self.hidden_size)),
        )
        return payload
