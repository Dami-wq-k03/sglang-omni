# CI Labels

SGLang-Omni CI runs on self-hosted runners: H100 machines for the GPU stages, an Intel Xeon host, an Intel GPU host and an Apple Silicon Mac. A pull request runs none of it until a maintainer adds CI labels. Once labeled, every later push re-runs CI for as long as the labels stay. Draft pull requests are blocked even when they are labeled.

There are two kinds of labels. A stage label decides which part of CI runs. A model label decides which model a stage tests. This page lists every stage, the jobs and models it covers, and what each label does. `.github/scripts/resolve_ci_labels.py` implements these rules, and every CI workflow resolves its labels through that script.

## Rules

1. A stage runs only when its stage label or `run-all-ci` is on the pull request. `run-all-ci` selects every stage.
2. A pull request needs at least one stage label. Without one, the CI gate fails with `No CI label found`.
3. Any stage label also runs the H100 unit tests (PR Test). This includes the hardware stages: a pull request labeled only `run-mac-ci` still runs PR Test on H100.
4. A model label only picks the model of a selected stage. It never selects a stage by itself. If neither its stage label nor `run-all-ci` is present, the label is ignored and the gate prints a warning.
5. A selected stage without a model label draws its model at random from that stage's pool. The draw is seeded by the workflow run ID, so re-running jobs in the same run keeps the same model, while a new push or a newly added label starts a new run and draws again.
6. Each stage takes at most one model label. Two model labels for the same stage fail the gate, even when that stage is not selected.
7. Stage labels combine. `run-tts-ci` together with `run-mps-ci` runs PR Test, TTS and MPS, and nothing else.

## Stage labels

| Label | Stage | Workflow | Runner |
| --- | --- | --- | --- |
| `run-all-ci` | Every stage below | All of them | All of them |
| `run-asr-ci` | ASR | Omni CI | H100 |
| `run-tts-ci` | TTS | Omni CI | H100 |
| `run-mps-ci` | MPS | Omni CI | H100 |
| `run-omni-ci` | Omni | Omni CI | H100 |
| `run-intel-cpu-ci` | Intel CPU | CPU CI | Intel Xeon (`intel-omni-spr`) |
| `run-intel-xpu-ci` | Intel XPU | XPU CI | Intel GPU (`intel-omni-bmg`) |
| `run-mac-ci` | Mac | Voxt Mac CI | Apple Silicon |

XPU means the Intel GPU runner. Huawei Ascend has no test CI.

## The Omni CI workflow

The Omni CI workflow hosts PR Test and the four H100 model stages. Its preflight job resolves the labels once. A shared setup job then builds the virtualenv and downloads model weights. The suites run in this order:

PR Test → ASR → TTS → MPS → Omni

Each suite waits for the previous one to finish. A suite that fails or is not selected does not stop the suites after it. A failed setup job blocks every suite.

### PR Test

PR Test has no label of its own and runs whenever any stage is selected. It has two jobs on H100:

| Job | What it runs |
| --- | --- |
| `unit-test` | `pytest tests/ -m "not benchmark and not accelerator"`, with CUDA hidden from the process |
| `unit-test-accelerator` | `pytest tests/ -m "accelerator and not benchmark"`, the unit tests that need a GPU |

### ASR (`run-asr-ci`)

| Job | What it covers |
| --- | --- |
| stage 1 - MOSS-TD multi-speaker | Multi-speaker transcription and diarization with MOSS-Transcribe-Diarize, behind a two-worker router: movies800 in offline and streaming mode, AISHELL-4 long-form audio and googletime. It gates character error rate and speed. This job always uses MOSS-Transcribe-Diarize, whatever model the stage draws. |
| stage 2 - SeedTTS ASR | The selected model on the SeedTTS English and Chinese splits. It gates word error rate on both splits, and throughput, latency and real-time factor on the English split. |

The selected model applies to stage 2:

| Model label | Checkpoint | In the random pool |
| --- | --- | --- |
| `run-asr-fun-asr-nano-2512` | `FunAudioLLM/Fun-ASR-Nano-2512-hf` | Yes |
| `run-asr-qwen3-asr-1.7b` | `Qwen/Qwen3-ASR-1.7B` | Yes |
| `run-asr-whisper-large-v3` | `openai/whisper-large-v3` | Yes |

### TTS (`run-tts-ci`)

| Job | What it covers |
| --- | --- |
| stage 1 - non-streaming | Voice cloning on SeedTTS English at concurrency 16, behind a two-worker router: speed, word error rate, speaker similarity and UTMOS. |
| stage 2 - streaming | The same benchmark with streaming output: speed and word error rate. |
| stage latency - streaming first audio | Median time to first playable audio under Poisson arrivals, with one worker alone on the GPU. Runs only when the stage uses one of the two Qwen3-TTS checkpoints. |
| stage 3 - consistency | Checks the artifacts of stages 1 and 2: both cover all 1088 SeedTTS English samples, every streaming request finished, and streaming and non-streaming audio durations agree. |
| stage 4 - serving stress | A mixed production workload over the REST, streaming and WebSocket endpoints through a two-worker router. This job always uses Higgs-TTS, whatever model the stage draws. |

The selected model applies to stages 1, 2, 3 and the latency stage:

| Model label | Checkpoint | In the random pool |
| --- | --- | --- |
| `run-tts-higgs-tts-3-4b` | `bosonai/higgs-tts-3-4b` | Yes |
| `run-tts-moss-tts-local-transformer-v1.5` | `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5` | Yes |
| `run-tts-qwen3-tts-12hz-1.7b-base` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Yes |
| `run-tts-qwen3-tts-12hz-1.7b-customvoice` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | No |
| `run-tts-fun-cosyvoice3-0.5b-2512` | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | Yes |

Qwen3-TTS CustomVoice uses a named voice instead of a reference clip, and its thresholds are not yet calibrated on the CI host. It reports its metrics without gating on them, and only its model label selects it.

### MPS (`run-mps-ci`)

MPS has one job, stage 1 - MPS non-streaming. It runs two TTS router workers on a single H100 that share the GPU through CUDA MPS, and checks three things: the two workers are separate MPS clients on the intended GPU, they actually execute at the same time, and the non-streaming word error rate, speaker similarity, UTMOS and audio integrity match the TTS stage's assertions.

MPS draws its model independently of the TTS stage, and TTS model labels do not affect it:

| Model label | Checkpoint | MPS config | In the random pool |
| --- | --- | --- | --- |
| `run-mps-higgs-tts-3-4b` | `bosonai/higgs-tts-3-4b` | `examples/mps_dp/configs/higgs_h100_dp3.yaml` | Yes |
| `run-mps-moss-tts-local-transformer-v1.5` | `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5` | `examples/mps_dp/configs/moss_local_h100_dp2.yaml` | Yes |

### Omni (`run-omni-ci`)

Every job runs the selected model unless noted otherwise. A job named "Talker" asks for text and audio output, and also gates the word error rate between the text answer and an ASR transcript of the spoken answer.

| Job | What it covers |
| --- | --- |
| stage 1 - thinker length integration | Overlong prompts and prompt plus `max_tokens` overflow return HTTP 400; decoding that hits `max_tokens` returns `finish_reason="length"`. |
| stage 2 - TTS | Voice cloning on SeedTTS: speed, word error rate, speaker similarity and UTMOS. |
| stages 3 and 4 - MMMU, MMMU Talker | Image and text question answering: accuracy and speed. |
| stages 5 and 6 - MMSU, MMSU Talker | Spoken audio understanding: accuracy and speed. |
| stages 7 and 8 - Video-MME, Video-MME Talker | Video understanding: accuracy and speed. |
| stage 9 - Video-AMME | Video with its audio track: accuracy and speed. |
| stage 10 - Video-AMME Talker | Video with audio and spoken answers, with the thinker sharded across two GPUs (tensor parallel size 2). |
| stage 11 - process replicas | A two-GPU deployment with one code2wav replica on each GPU; checks that both replicas start, register and receive requests. Runs only for Qwen3-Omni. |

| Model label | Checkpoint | In the random pool |
| --- | --- | --- |
| `run-omni-qwen3-omni-30b-a3b-instruct` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | Yes |
| `run-omni-minicpm-o-4.5` | `openbmb/MiniCPM-o-4_5` | Yes |

## Hardware stages

The hardware stages are separate workflows with no model labels. Each keeps a path filter, so a stage label alone does not force a run. A pull request that changes none of the stage's paths skips its tests.

### Intel CPU (`run-intel-cpu-ci`)

The CPU CI workflow builds the Intel Xeon image from `docker/cpu.Dockerfile` and runs `pytest tests/unit_test/cpu/`. It runs when the pull request touches `sglang_omni/` (except Markdown files), `tests/unit_test/cpu/`, `pyproject_cpu.toml`, `scripts/cpu/`, `docker/cpu.Dockerfile` or the workflow itself. When those paths are untouched, or when the pull request selects other stages but not Intel CPU, the test job is skipped and the `CPU CI Status` check still reports success.

### Intel XPU (`run-intel-xpu-ci`)

The XPU CI workflow pulls or builds the Intel GPU image from `docker/xpu.Dockerfile` and runs the device layer tests in `tests/unit_test/xpu/test_device_layer.py` and the dots.tts flow head and tail tests. It runs when the pull request touches `sglang_omni/`, `tests/unit_test/xpu/`, `tests/unit_test/dots_tts/`, `pyproject_xpu.toml`, `scripts/xpu/`, `docker/xpu.Dockerfile` or the workflow itself.

### Mac (`run-mac-ci`)

The Voxt Mac CI workflow has two jobs on Apple Silicon. The native runtime job builds the MLX runtime, checks its golden outputs on a frozen corpus for Qwen3-ASR and the other native models, and runs the server API tests. The Voxt unit test job builds the Voxt app with Xcode and runs its Omni and MLX test classes. The workflow starts only when the pull request touches `Voxt/`, `sglang_omni_mlx/` or the workflow itself; otherwise it does not appear on the pull request at all.

## Examples

| Labels | What runs |
| --- | --- |
| `run-tts-ci` | PR Test, then TTS on a random model from its pool. |
| `run-tts-ci`, `run-tts-moss-tts-local-transformer-v1.5` | PR Test, then TTS on MOSS-TTS. |
| `run-tts-ci`, `run-mps-ci` | PR Test, TTS and MPS, each with its own random model. |
| `run-mac-ci` | PR Test on H100 and Voxt Mac CI. No H100 model stage. |
| `run-all-ci` | Every stage. Each model stage draws its own model. |
| `run-all-ci`, `run-omni-minicpm-o-4.5` | Every stage. Omni uses MiniCPM-o 4.5, the other stages draw at random. |
| `run-tts-ci`, `run-omni-minicpm-o-4.5` | PR Test and TTS. The Omni model label is ignored with a warning, because Omni is not selected. |
| `run-omni-minicpm-o-4.5` | Nothing. The gate fails, because a model label does not select a stage. |
| `run-tts-ci`, `run-tts-higgs-tts-3-4b`, `run-tts-moss-tts-local-transformer-v1.5` | Nothing. The gate fails, because TTS has two model labels. |

## Label changes, re-runs and manual runs

Adding a label, CI or not, starts a new run of Omni CI, CPU CI, XPU CI and Voxt Mac CI (when its paths match) and cancels the run of the same workflow in progress, so the new label set takes effect at once. Models that no label pins are drawn again in the new run.

Removing a label cancels only the workflows that the remaining labels no longer select. Removing `run-mac-ci` while `run-tts-ci` stays cancels Voxt Mac CI and leaves Omni CI running. Removing the last stage label cancels Omni CI, CPU CI, XPU CI and Voxt Mac CI. A run that is still selected keeps the stages and models it resolved when it started; push again or re-run all jobs to pick up the change.

Re-running jobs reuses the run ID, so every stage keeps the model it drew. The gate reads the live labels from the pull request, so re-running all jobs picks up labels added or removed since the run started.

A maintainer can start Omni CI by hand from the Actions tab or with `gh workflow run`. The `ci_labels` input takes space-separated labels and resolves them exactly like pull request labels; it defaults to `run-all-ci`:

```bash
gh workflow run omni-ci.yaml --ref <branch> \
  -f ci_labels="run-tts-ci run-tts-moss-tts-local-transformer-v1.5"
```

Manual runs of CPU CI, XPU CI and Voxt Mac CI ignore labels and always run their tests.

The `/rerun-failed-ci` comment command still re-runs the failed jobs of a pull request. The `/tag-run-ci-label` and `/tag-and-rerun-ci` commands are disabled while they are redesigned around these labels; the bot replies that they are disabled.

## Gate messages

| Message | Meaning |
| --- | --- |
| `No CI label found. Add run-all-ci or a stage label: ...` | The pull request has no stage label. Add `run-all-ci` or a stage label. |
| `Keep only one <stage> model label: ...` | Two model labels name the same stage. Remove one of them. |
| `Ignoring <label>: its stage label and run-all-ci are both missing.` | A warning, not a failure. The model label has no effect until its stage is selected. |

A branch created before these labels existed does not contain `resolve_ci_labels.py`. On such a branch, the Omni CI and Voxt Mac CI gates fail until the branch merges the current `main`.
