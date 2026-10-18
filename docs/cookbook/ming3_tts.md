# Ming-3.0-tts-tiny

[Ming-3.0-tts-tiny](https://huggingface.co/inclusionAI/Ming-3.0-tts-tiny) synthesizes speech from text and can clone one reference voice. A Ling V3 mixture-of-experts backbone predicts one continuous 64-dimensional latent per step through a flow-matching head. A CAM++ speaker embedding conditions the prompt, and an AudioVAE decoder turns the latents into a 44.1 kHz waveform.

## Installation and Startup

Install SGLang-Omni by following [Installation](../get_started/installation.md).

Startup uses one GPU. Preprocessing runs on the CPU. The backbone and the VAE decoder share that GPU.

```bash
sgl-omni serve --model-path inclusionAI/Ming-3.0-tts-tiny
```

## Speech synthesis

```python
import base64
from pathlib import Path

import requests

reference = base64.b64encode(Path("reference.wav").read_bytes()).decode()
response = requests.post(
    "http://127.0.0.1:8000/v1/audio/speech",
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

The response body is one complete waveform. Streaming output is not implemented. The checkpoint conditions only on the speaker embedding, so `ref_text` is accepted and ignored. Omit `ref_audio` to use a zero speaker embedding. A request accepts at most one reference.

`temperature` overrides the checkpoint default of 1.5. `seed` makes the flow-matching noise reproducible. `max_new_tokens` cannot exceed the checkpoint's 250 latent steps. Speech stops when the stop head exceeds 0.5, and not before 13 steps.
