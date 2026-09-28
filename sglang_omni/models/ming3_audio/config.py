# SPDX-License-Identifier: Apache-2.0
"""Ming 3 audio pipeline configuration."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config.schema import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    PlacementConfig,
    StageConfig,
)

PACKAGE = "sglang_omni.models.ming3_audio"


def default_stages() -> list[StageConfig]:
    return [
        StageConfig(
            name="preprocessing",
            process="preprocessing",
            factory_path=f"{PACKAGE}.stages.create_preprocessing_executor",
            next="audio_encoder",
        ),
        StageConfig(
            name="audio_encoder",
            process="audio_encoder",
            gpu=0,
            factory_path=f"{PACKAGE}.stages.create_audio_encoder_executor",
            next="thinker",
        ),
        EngineStageConfig(
            name="thinker",
            process="thinker",
            gpu=[0, 1],
            tp_size=2,
            factory_path=f"{PACKAGE}.stages.create_thinker_executor",
            factory=FactoryArgs(max_seq_len=8192, enable_async_decode=False),
            engine=EngineArgs(
                mem_fraction_static=0.93,
                max_running_requests=8,
                disable_cuda_graph=True,
                enable_torch_compile=False,
            ),
            terminal=True,
        ),
    ]


class Ming3AudioPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "BailingMoeV3AudioForConditionalGeneration"
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "thinker": EngineStageConfig,
    }
    model_path: str
    entry_stage: str = "preprocessing"
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )
    stages: list[StageConfig] = Field(default_factory=default_stages)


EntryClass = Ming3AudioPipelineConfig
