# SPDX-License-Identifier: Apache-2.0
"""Stage factories for Ming 3 audio understanding."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from pydantic import JsonValue
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.sampling.sampling_params import SamplingParams
from transformers import AutoConfig, AutoTokenizer

from sglang_omni.models.ming3_audio.audio_encoder import (
    AudioEncoderConfig,
    Ming3AudioEncoder,
)
from sglang_omni.models.ming3_audio.hf_config import Ming3AudioConfig
from sglang_omni.models.ming3_audio.preprocessing import (
    AudioPipelineState,
    Ming3AudioPreprocessor,
)
from sglang_omni.models.ming3_audio.thinker import (
    Ming3AudioModelRunner,
    Ming3AudioThinker,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.sglang_backend.server_args_builder import (
    build_sglang_server_args,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.device import resolve_concrete_device


def create_preprocessing_executor(model_path: str) -> SimpleScheduler:
    return SimpleScheduler(Ming3AudioPreprocessor(model_path))


def create_audio_encoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
) -> SimpleScheduler:
    configuration = json.loads((Path(model_path) / "config.json").read_text())
    encoder = Ming3AudioEncoder(
        AudioEncoderConfig.model_validate(configuration["audio_config"]),
        configuration["text_config"]["hidden_size"],
    )
    encoder.load_checkpoint(model_path)
    encoder = encoder.eval().to(
        resolve_concrete_device(device, gpu_id), dtype=torch.bfloat16
    )

    @torch.inference_mode()
    def encode(payload: StagePayload) -> StagePayload:
        state: AudioPipelineState = payload.data
        if state["features"]:
            state["audio_embeddings"] = encoder(state["features"]).cpu()
        else:
            pass
        state["features"] = []
        return payload

    return SimpleScheduler(encode)


def create_thinker_executor(
    model_path: str,
    *,
    max_seq_len: int,
    enable_async_decode: bool,
    device: str | None = None,
    gpu_id: int | None = None,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
    server_args_overrides: dict[str, JsonValue] | None = None,
    total_gpu_memory_fraction: float | None = None,
) -> OmniScheduler:
    AutoConfig.register(Ming3AudioConfig.model_type, Ming3AudioConfig, exist_ok=True)
    ModelRegistry.models["BailingMoeV3ForCausalLM"] = Ming3AudioThinker
    overrides = dict(server_args_overrides or {})
    overrides.update(
        tp_size=tp_size,
        trust_remote_code=False,
        json_model_override_args=json.dumps({"architectures": ["BailingMoeV3ForCausalLM"]}),
        disable_radix_cache=True,
        chunked_prefill_size=-1,
    )
    server_args = build_sglang_server_args(
        model_path, context_length=max_seq_len, **overrides
    )
    worker, tree_cache, request_pool, token_pool, model_config = (
        create_sglang_infrastructure(
            server_args,
            resolve_concrete_device(device, gpu_id).index or 0,
            tp_rank=tp_rank,
            nccl_port=nccl_port,
            total_gpu_memory_fraction=total_gpu_memory_fraction,
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)

    def build_request(payload: StagePayload) -> SGLangARRequestData:
        state: AudioPipelineState = payload.data
        parameters = payload.request.params
        sampling = SamplingParams(
            temperature=parameters.get("temperature", 0.0),
            max_new_tokens=parameters.get("max_new_tokens", 512),
            top_p=parameters.get("top_p", 1.0),
            top_k=parameters.get("top_k", -1),
            repetition_penalty=parameters.get("repetition_penalty", 1.0),
            sampling_seed=parameters.get("seed"),
        )
        sampling.normalize(tokenizer)
        sampling.verify(model_config.vocab_size)
        request = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=state["input_ids"].tolist(),
            sampling_params=sampling,
            vocab_size=model_config.vocab_size,
            eos_token_ids={tokenizer.eos_token_id},
        )
        request.tokenizer = tokenizer
        return SGLangARRequestData(
            input_ids=state["input_ids"],
            req=request,
            model_inputs={
                "audio_positions": state["audio_positions"],
                "audio_embeddings": state["audio_embeddings"],
            },
            max_new_tokens=sampling.max_new_tokens,
            temperature=sampling.temperature,
            output_ids=request.output_ids,
            stage_payload=payload,
        )

    def build_result(request: SGLangARRequestData) -> StagePayload:
        payload = request.stage_payload
        payload.data = {
            "text": tokenizer.decode(request.output_ids, skip_special_tokens=True),
            "usage": {
                "prompt_tokens": len(request.input_ids),
                "completion_tokens": len(request.output_ids),
            },
        }
        return payload

    processor = SGLangOutputProcessor()
    return OmniScheduler(
        tp_worker=worker,
        tree_cache=tree_cache,
        req_to_token_pool=request_pool,
        token_to_kv_pool_allocator=token_pool,
        server_args=server_args,
        model_config=model_config,
        model_runner=Ming3AudioModelRunner(worker, processor),
        request_builder=build_request,
        result_adapter=build_result,
        enable_async_decode=enable_async_decode,
    )
