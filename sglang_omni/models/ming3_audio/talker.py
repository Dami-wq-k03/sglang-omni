# SPDX-License-Identifier: Apache-2.0
"""Continuous autoregressive speech synthesis for Ming 3 audio."""

from __future__ import annotations

import io
import json
from pathlib import Path

import onnxruntime
import soundfile
import torch
import torch.nn.functional as functional
import torchaudio
from pydantic import BaseModel, JsonValue
from safetensors.torch import load_file
from torch import nn
from transformers import (
    AutoTokenizer,
    DynamicCache,
    PreTrainedTokenizerBase,
    Qwen2Config,
    Qwen2Model,
)

from sglang_omni.modules.ming_audio.flow.aggregator import Aggregator
from sglang_omni.modules.ming_audio.flow.cfm import CFM, get_epss_timesteps
from sglang_omni.modules.ming_audio.flow.dit import DiT
from sglang_omni.modules.ming_audio.vae.configuration_audio_vae import AudioVAEconfig
from sglang_omni.modules.ming_audio.vae.modeling_audio_vae import AudioVAE
from sglang_omni.utils.audio_features import cached_fbank

TTS_PROMPT = "Please generate speech based on the following description.\n"
MINIMUM_DECODE_STEPS = 3
SPEAKER_SAMPLE_RATE = 16000


class TalkerConfig(BaseModel):
    aggregator: dict[str, JsonValue]
    flowmodel: dict[str, JsonValue]
    patch_size: int
    history_patch_size: int
    steps: int
    cfg_strength: float


class Ming3Talker(nn.Module):
    def __init__(self, config: TalkerConfig, language_config: Qwen2Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen2Model(language_config)
        self.cfm = CFM(
            DiT(llm_cond_dim=language_config.hidden_size, **config.flowmodel),
            steps=config.steps,
        )
        self.aggregator = Aggregator(
            llm_input_dim=language_config.hidden_size, **config.aggregator
        )
        self.stop_head = nn.Linear(language_config.hidden_size, 2)
        self.spk_head = nn.Linear(192, language_config.hidden_size)

    def generate_latents(
        self,
        embeddings: torch.Tensor,
        reference_latents: torch.Tensor,
        max_decode_steps: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        config = self.config
        history = functional.pad(
            reference_latents[:, -config.history_patch_size :],
            (0, 0, max(0, config.history_patch_size - reference_latents.shape[1]), 0),
        )
        timesteps = get_epss_timesteps(
            config.steps, embeddings.device, embeddings.dtype
        )
        diffusion_parameters = embeddings.new_tensor([config.cfg_strength, 0.25, 0.0])
        cache = DynamicCache(config=self.model.config)
        generated: list[torch.Tensor] = []
        for step in range(max_decode_steps):
            hidden = self.model(
                inputs_embeds=embeddings, past_key_values=cache, use_cache=True
            ).last_hidden_state[:, -1:]
            noise = torch.randn(
                (1, config.patch_size, history.shape[-1]),
                device=embeddings.device,
                dtype=embeddings.dtype,
                generator=generator,
            )
            diffusion_noise = torch.randn(
                (config.steps, *noise.shape),
                device=noise.device,
                dtype=noise.dtype,
                generator=generator,
            )
            latent = self.cfm.sample(
                hidden, history, noise, timesteps, diffusion_parameters, diffusion_noise
            )
            embeddings = self.aggregator(latent)
            history = torch.cat([history[:, config.patch_size :], latent], dim=1)
            generated.append(latent)
            stop_probability = self.stop_head(hidden[:, -1]).softmax(dim=-1)[0, 1]
            if step > MINIMUM_DECODE_STEPS and bool(stop_probability > 0.5):
                break
            else:
                pass
        return torch.cat(generated, dim=1)


class Ming3SpeechGenerator:
    def __init__(
        self, model_path: str, device: torch.device, max_decode_steps: int
    ) -> None:
        path = Path(model_path) / "talker"
        configuration = TalkerConfig.model_validate_json(
            (path / "config.json").read_text()
        )
        language_config = Qwen2Config.from_pretrained(
            path / "llm", attn_implementation="sdpa"
        )
        self.talker = Ming3Talker(configuration, language_config).eval()
        self.talker.load_state_dict(load_file(str(path / "model.safetensors")))
        self.talker.to(device=device, dtype=torch.bfloat16)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            path / "llm", trust_remote_code=False
        )
        vae_configuration = json.loads((path / "vae/config.json").read_text())
        for name in ("enc_kwargs", "dec_kwargs"):
            vae_configuration[name]["backbone"]["_attn_implementation"] = "sdpa"
        self.vae = AudioVAE(AudioVAEconfig(**vae_configuration)).eval()
        self.vae.load_state_dict(load_file(str(path / "vae/model.safetensors")))
        self.vae.to(device=device, dtype=torch.bfloat16)
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 2
        self.speaker_encoder = onnxruntime.InferenceSession(
            str(path / "campplus.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.device = device
        self.sample_rate: int = vae_configuration["sample_rate"]
        self.max_decode_steps = max_decode_steps

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        reference_audio: bytes | None,
        reference_text: str,
        seed: int,
    ) -> torch.Tensor:
        talker = self.talker
        tokenizer = self.tokenizer
        latent_channels = talker.aggregator.in_channels
        reference_latents = torch.empty(
            (1, 0, latent_channels), device=self.device, dtype=torch.bfloat16
        )
        reference_embeddings = torch.empty(
            (1, 0, talker.model.config.hidden_size),
            device=self.device,
            dtype=torch.bfloat16,
        )
        speaker_embedding: torch.Tensor | None = None
        if reference_audio is not None:
            waveform, sample_rate = soundfile.read(
                io.BytesIO(reference_audio), dtype="float32", always_2d=True
            )
            waveform = torch.from_numpy(waveform.mean(axis=1)).unsqueeze(0)
            speaker_waveform = torchaudio.functional.resample(
                waveform, sample_rate, SPEAKER_SAMPLE_RATE
            )
            features = cached_fbank(
                speaker_waveform, num_mel_bins=80, sample_frequency=SPEAKER_SAMPLE_RATE
            )
            features -= features.mean(dim=0, keepdim=True)
            speaker_vector = self.speaker_encoder.run(
                None,
                {
                    self.speaker_encoder.get_inputs()[0]
                    .name: features.unsqueeze(0)
                    .numpy()
                },
            )[0]
            speaker_embedding = talker.spk_head(
                torch.from_numpy(speaker_vector).to(self.device, dtype=torch.bfloat16)
            )
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.sample_rate
            )
            samples_per_patch = (
                self.vae.encoder.hop_size
                * self.vae.encoder.patch_size
                * talker.config.patch_size
            )
            waveform = functional.pad(
                waveform, ((-waveform.shape[-1]) % samples_per_patch, 0)
            )
            # note (Codex): VAE posterior sampling consumes the reference RNG before AR decoding.
            with torch.random.fork_rng(devices=[self.device]):
                torch.manual_seed(seed)
                reference_latents, _ = self.vae.encode_latent(
                    waveform.to(self.device, dtype=torch.bfloat16),
                    torch.tensor([waveform.shape[-1]], device=self.device),
                )
            patches = reference_latents.reshape(
                -1, talker.config.patch_size, latent_channels
            )
            reference_embeddings = talker.aggregator(patches).reshape(
                1, -1, talker.model.config.hidden_size
            )
        else:
            pass

        speaker_prompt = (
            "  speaker_1:<|vision_start|><|vision_pad|><|vision_end|>\n"
            if speaker_embedding is not None
            else ""
        )
        prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{TTS_PROMPT}{speaker_prompt} Text input:\n"
            f"{reference_text}{text}<|im_end|>\n<|im_start|>assistant\n<audio>"
        )
        prompt_ids = tokenizer.encode(prompt)
        reference_count = reference_embeddings.shape[1] if reference_text else 0
        prompt_ids.extend(
            [tokenizer.convert_tokens_to_ids("<audioPatch>")] * reference_count
        )
        token_ids = torch.tensor([prompt_ids], device=self.device)
        embeddings = talker.model.embed_tokens(token_ids)
        if speaker_embedding is not None:
            speaker_position = (
                prompt_ids.index(tokenizer.convert_tokens_to_ids("<|vision_start|>"))
                + 1
            )
            embeddings[0, speaker_position] = speaker_embedding[0]
        else:
            pass
        if reference_count:
            embeddings[:, -reference_count:] = reference_embeddings
        else:
            pass
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = talker.generate_latents(
            embeddings, reference_latents, self.max_decode_steps, generator
        )
        waveform, _, _ = self.vae.decode(latents, use_cache=False, last_chunk=True)
        return waveform.flatten().float().cpu()
