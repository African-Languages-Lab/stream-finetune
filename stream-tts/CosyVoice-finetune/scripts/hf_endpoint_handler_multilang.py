"""
Multi-language HF Inference Endpoint handler: the orchestrator repo itself holds no
per-language weights -- at container startup it pulls each language's llm.pt/flow.pt straight
from its own cosyvoice3-individual-{lang} repo via the HF Hub API, so each language has exactly
one copy of its weights (that repo), not a duplicate baked into this orchestrator too. Shared
assets (tokenizer/campplus/speech_tokenizer/original vocoder) live once, physically, in this
repo, since they're identical across every language repo already.

Requires an HF token with read access to the (private) LANGUAGE_REPOS at runtime -- set it as
an endpoint secret/environment variable named HF_TOKEN when deploying (Inference Endpoints ->
this endpoint -> Settings -> Environment variables). Without it, download_language() below will
fail for every language and the endpoint will come up with an empty self.models.

Adding a language later: push its weights to a new all-lab/cosyvoice3-individual-{code} repo
(build_and_push_proven_hf_bundles.py), add {code} to LANGUAGE_REPOS below, redeploy this
orchestrator repo (handler.py changed) and restart the endpoint -- no weights re-uploaded here.

Request contract (same zero-shot cloning pattern as every synthesis script in this project):
{
  "inputs": "text to synthesize",
  "parameters": {
    "language": "ha-NG",
    "prompt_text": "exact transcript of the reference clip",
    "prompt_audio_base64": "<base64-encoded wav bytes, mono, any samplerate>"
  }
}
Response: {"language", "audio_base64", "sample_rate", "duration_sec", "peak"} or {"error": ...}.
"""
import base64
import os
import sys
import tempfile

import torch
import torchaudio
from huggingface_hub import hf_hub_download

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "third_party", "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import CosyVoice3  # noqa: E402

SHARED_ASSETS = ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "CosyVoice-BlankEN", "hift.pt"]
ORG = "all-lab"
LANGUAGE_REPOS = {
    "ha-NG": f"{ORG}/cosyvoice3-individual-ha-NG",
    "tw-GH": f"{ORG}/cosyvoice3-individual-tw-GH",
    "ig-NG": f"{ORG}/cosyvoice3-individual-ig-NG",
    "ee-GH": f"{ORG}/cosyvoice3-individual-ee-GH",
}


class EndpointHandler:
    def __init__(self, path=""):
        root = path or _HERE
        token = os.environ.get("HF_TOKEN")
        self._workdir = tempfile.mkdtemp(prefix="cosyvoice_multilang_")
        self.models = {}

        for lang, repo_id in LANGUAGE_REPOS.items():
            lang_dir = os.path.join(self._workdir, lang)
            os.makedirs(lang_dir, exist_ok=True)
            for asset in SHARED_ASSETS:
                dst = os.path.join(lang_dir, asset)
                if not os.path.exists(dst):
                    os.symlink(os.path.join(root, asset), dst)

            try:
                print(f"[handler] downloading '{lang}' weights from {repo_id}...", flush=True)
                for f in ("llm.pt", "flow.pt"):
                    downloaded = hf_hub_download(repo_id=repo_id, filename=f, token=token)
                    os.symlink(downloaded, os.path.join(lang_dir, f))
                print(f"[handler] loading '{lang}'...", flush=True)
                self.models[lang] = CosyVoice3(lang_dir, fp16=False)
            except Exception as e:
                print(f"[handler] FAILED to load '{lang}' from {repo_id}: {e}", flush=True)

        print(f"[handler] ready, languages={sorted(self.models.keys())}", flush=True)

    def __call__(self, data):
        params = data.get("parameters") or {}
        language = params.get("language")
        if language not in self.models:
            return {"error": f"unknown or missing 'language' parameter, available: {sorted(self.models.keys())}"}

        target_text = data.get("inputs")
        prompt_text = params.get("prompt_text")
        prompt_audio_b64 = params.get("prompt_audio_base64")
        if not target_text or not prompt_text or not prompt_audio_b64:
            return {"error": "requires inputs (target text), parameters.language, parameters.prompt_text, parameters.prompt_audio_base64"}

        if not prompt_text.endswith("<|endofprompt|>"):
            prompt_text = prompt_text + "<|endofprompt|>"

        model = self.models[language]
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(base64.b64decode(prompt_audio_b64))
            tmp.flush()

            audio, last_err = None, None
            for _ in range(4):
                try:
                    results = list(model.inference_zero_shot(target_text, prompt_text, tmp.name, stream=False))
                    audio = results[0]["tts_speech"]
                    break
                except RuntimeError as e:
                    last_err = e
            if audio is None:
                return {"error": f"synthesis failed after retries: {last_err}"}

        out_path = tempfile.mktemp(suffix=".wav")
        torchaudio.save(out_path, audio, model.sample_rate)
        with open(out_path, "rb") as f:
            audio_bytes = f.read()
        os.remove(out_path)

        return {
            "language": language,
            "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
            "sample_rate": model.sample_rate,
            "duration_sec": audio.shape[1] / model.sample_rate,
            "peak": audio.abs().max().item(),
        }
