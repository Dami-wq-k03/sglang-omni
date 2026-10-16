# Ming 3 TTS

Ming 3 TTS synthesizes speech from text, optionally cloning the voice of one
reference recording. The pipeline runs a Ling V3 mixture-of-experts backbone that
predicts one continuous 64-dimensional latent per step through a flow-matching
head. A CAM++ speaker embedding conditions the prompt, and an AudioVAE decoder
turns the latents into a 44.1 kHz waveform.

## Run locally

The default placement uses one GPU. Preprocessing runs on the CPU. The backbone
and the VAE decoder share GPU 0.

```bash
CUDA_VISIBLE_DEVICES=0 sgl-omni serve \
  --model-path inclusionAI/Ming-3.0-tts-tiny --host 127.0.0.1 --port 8011
```

Synthesize with a reference voice:

```python
import base64
from pathlib import Path

import requests

reference = base64.b64encode(Path("reference.wav").read_bytes()).decode()
response = requests.post(
    "http://127.0.0.1:8011/v1/audio/speech",
    json={
        "model": "inclusionAI/Ming-3.0-tts-tiny",
        "input": "Get the trust fund to the bank early.",
        "ref_audio": f"data:audio/wav;base64,{reference}",
        "response_format": "wav",
        "seed": 123,
    },
    timeout=180,
)
response.raise_for_status()
Path("speech.wav").write_bytes(response.content)
```

The checkpoint conditions only on the speaker embedding of the reference
recording, so `ref_text` is accepted but unused. Without a reference, a zero
speaker embedding is used. A request accepts at most one reference.

The checkpoint's `tts_config` defines the sampling defaults. Temperature defaults
to 1.5, and `temperature` overrides it. `max_new_tokens` is capped at the
checkpoint's 250 latent steps. Speech stops when the stop head exceeds 0.5,
after at least 13 steps. `seed` makes the flow-matching noise reproducible. The
pipeline returns the complete waveform; streaming output is not implemented.
Unsupported checkpoint contracts fail at load. These include a prompt format
other than `legacy_antmmf`, `with_rml=false`, patch or history sizes other than
one, and a decode alignment other than `old_shift`.

## Correctness baseline

Validation pins model revision `fe39337f33061c39b8863eb101e6754df257892c` and
SeedTTS dataset revision `27f4c1adee83b5b29b7c4b375f6b976324bda308`. This
dataset's English split contains **1088** entries. Each entry was synthesized
with its reference recording at seed 123 and concurrency 16 on one H200.
Qwen3-ASR-1.7B scored the output with the benchmark's English normalization.

The default configuration completed all 1088 entries without failures. It
measured **1.18% WER**, below the 1.567% reported on the model card. Mean latency
was 2.60 s for 3.64 s of audio, and throughput was 6.12 requests per second.

CAM++ checked the speaker conditioning on 200 random outputs. Each output's mean
cosine similarity was **0.830** to its own reference and **0.327** to another
entry's reference. Every output was closer to its own reference.

```bash
python -m pytest tests/unit_test/ming3_tts -q
python -m benchmarks.eval.benchmark_tts_seedtts --use-existing-server \
  --port 8011 --model inclusionAI/Ming-3.0-tts-tiny --lang en --seed 123 \
  --concurrency 16 --warmup 2 --generate-only --output-dir results/ming3_tts
python -m benchmarks.eval.benchmark_tts_seedtts --transcribe-only --lang en \
  --model inclusionAI/Ming-3.0-tts-tiny --port 8092 --output-dir results/ming3_tts
```

## Optimizations and parity

Each decode step replays a CUDA graph for the flow-matching sampler, the
aggregator that projects the latent back into the backbone, and the stop head.
Graphs are captured for power-of-two batch sizes up to `max_running_requests`.
Per-request noise is drawn outside the graph from each request's generator. With
the backbone in eager mode, the graph and eager versions of this step produced
bitwise-identical waveforms for the same seed.

Backbone decode also uses SGLang CUDA graphs. Each request's feedback embedding
is written to a fixed table, and decode reads it by row id through `input_ids`.
This table lets the captured graph replay feedback that is not a token. Graph
and eager backbones differ numerically and diverge under stochastic sampling, so
waveforms are not identical. On the full English split, the eager backbone
measured **1.17% WER** with 3.95 s mean latency. The graph backbone measured
**1.18% WER** with 2.60 s mean latency.

To reproduce the eager backbone, write a local JSON configuration and pass its
path to `sgl-omni serve --config`:

```json
{
  "config_cls": "Ming3TTSPipelineConfig",
  "model_path": "inclusionAI/Ming-3.0-tts-tiny",
  "stages": {
    "tts_engine": {"engine": {"disable_cuda_graph": true}}
  }
}
```

Set `stages.tts_engine.factory.enable_acoustic_graph` to `false` to run the
acoustic step eagerly as well.
