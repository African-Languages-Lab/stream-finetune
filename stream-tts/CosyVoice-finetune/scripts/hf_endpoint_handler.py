"""
Template for the HF Inference Endpoints custom handler ("handler.py") copied into each proven
model repo by push_hf_endpoint_files.py, alongside a vendored copy of cosyvoice/ and
third_party/Matcha-TTS/ (neither is a pip-installable package -- every script in this project
uses it via sys.path.insert against a local checkout, so vendoring is the only reliable way to
make each repo self-contained inside the endpoint container).

Request contract (matches the zero-shot cloning pattern used throughout this project --
CosyVoice3 has no reference-free/voiceless mode):
{
  "inputs": "text to synthesize",
  "parameters": {
    "prompt_text": "exact transcript of the reference clip",
    "prompt_audio_base64": "<base64-encoded wav bytes, mono, any samplerate>"
  }
}

Response:
{
  "audio_base64": "<base64-encoded wav bytes>",
  "sample_rate": 24000,
  "duration_sec": 3.45,
  "peak": 0.62
}

NOTE: not yet tested inside an actual HF Inference Endpoint container (that requires a live,
billed endpoint to verify against) -- the import surface and inference call mirror exactly what
every synthesis script in this project already does successfully, but if the container is
missing a transitive dependency on first deploy, check the endpoint's build/runtime logs and
add the missing package to requirements.txt.
"""
import base64
import os
import sys
import tempfile

import torch
import torchaudio

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "third_party", "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import CosyVoice3  # noqa: E402


class EndpointHandler:
    def __init__(self, path=""):
        model_dir = path or _HERE
        self.model = CosyVoice3(model_dir, fp16=False)

    def __call__(self, data):
        target_text = data.get("inputs")
        params = data.get("parameters") or {}
        prompt_text = params.get("prompt_text")
        prompt_audio_b64 = params.get("prompt_audio_base64")
        if not target_text or not prompt_text or not prompt_audio_b64:
            return {"error": "requires inputs (target text), parameters.prompt_text, parameters.prompt_audio_base64"}

        if not prompt_text.endswith("<|endofprompt|>"):
            prompt_text = prompt_text + "<|endofprompt|>"

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(base64.b64decode(prompt_audio_b64))
            tmp.flush()

            last_err = None
            for _ in range(4):
                try:
                    results = list(self.model.inference_zero_shot(target_text, prompt_text, tmp.name, stream=False))
                    audio = results[0]["tts_speech"]
                    break
                except RuntimeError as e:
                    last_err = e
                    audio = None
            if audio is None:
                return {"error": f"synthesis failed after retries: {last_err}"}

        out_path = tempfile.mktemp(suffix=".wav")
        torchaudio.save(out_path, audio, self.model.sample_rate)
        with open(out_path, "rb") as f:
            audio_bytes = f.read()
        os.remove(out_path)

        return {
            "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
            "sample_rate": self.model.sample_rate,
            "duration_sec": audio.shape[1] / self.model.sample_rate,
            "peak": audio.abs().max().item(),
        }
