# SPDX-License-Identifier: Apache-2.0
"""Single-stage native image generation pipeline for SenseNova-U1.5."""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from sglang_omni.config.schema import (
    FactoryArgs,
    FactoryDefault,
    PipelineConfig,
    StageConfig,
)

DEFAULT_GENERATION_DTYPE = "bfloat16"
DEFAULT_LAYERWISE_OFFLOAD = False


class SenseNovaGenerationFactoryArgs(FactoryArgs):
    dtype: Annotated[str | None, FactoryDefault(value=DEFAULT_GENERATION_DTYPE)] = None
    enable_cache_dit: bool | None = None
    cache_dit_params: dict[str, int | float] | None = None
    dit_layerwise_offload: Annotated[
        bool | None, FactoryDefault(value=DEFAULT_LAYERWISE_OFFLOAD)
    ] = None
    dit_offload_prefetch_size: Annotated[int | None, FactoryDefault(value=1)] = Field(
        default=None, ge=0, strict=True
    )
    dit_layerwise_resident_layers: Annotated[int | None, FactoryDefault(value=0)] = (
        Field(default=None, ge=0, strict=True)
    )
    dit_layerwise_residency_policy: Annotated[
        Literal["leading", "strided"] | None, FactoryDefault(value="leading")
    ] = None
    pin_cpu_memory: Annotated[bool | None, FactoryDefault(value=False)] = None


class SenseNovaGenerationStageConfig(StageConfig):
    factory: SenseNovaGenerationFactoryArgs = Field(
        default_factory=SenseNovaGenerationFactoryArgs
    )


class SenseNovaU1PipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "NEOChatModel"
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "generate": SenseNovaGenerationStageConfig,
    }

    model_path: str
    entry_stage: str = "generate"
    stages: list[StageConfig] = [  # noqa: RUF012 -- Pydantic model default
        SenseNovaGenerationStageConfig(
            name="generate",
            process="sensenova_generate",
            factory_path="sglang_omni.models.sensenova_u1.stages.create_generation_executor",
            factory=SenseNovaGenerationFactoryArgs(
                dtype=DEFAULT_GENERATION_DTYPE,
                dit_layerwise_offload=DEFAULT_LAYERWISE_OFFLOAD,
                enable_cache_dit=False,
            ),
            gpu=0,
            terminal=True,
        )
    ]

    def model_post_init(self, __context: object = None, /) -> None:
        super().model_post_init(__context)
        generate = next(stage for stage in self.stages if stage.name == "generate")
        if generate.tp_size not in (1, 2, 4, 8):
            raise ValueError("SenseNova TP supports tp_size=1, 2, 4, or 8")
        else:
            pass
        if generate.tp_size > 1 and (generate.factory.max_batch_size or 1) != 1:
            raise ValueError("SenseNova TP requires max_batch_size=1")
        else:
            pass

        replicas = self.processes.get(generate.process or generate.name)
        if replicas is not None and replicas.num_replicas > 1:
            devices = replicas.replica_devices
            if devices is not None and len(set(devices)) != len(devices):
                raise ValueError(
                    "SenseNova-U1 DP requires distinct GPUs across replicas and TP ranks"
                )
            else:
                pass
        else:
            pass


EntryClass = SenseNovaU1PipelineConfig
