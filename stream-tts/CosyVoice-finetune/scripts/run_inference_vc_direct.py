"""
Real speech -> flow -> hifigan, entirely skipping llm/text. CosyVoice ships this as a
first-class feature (voice conversion): inference_vc(source_wav, prompt_wav) extracts
source_wav's own discrete speech tokens (via the frozen speech tokenizer, no LLM), feeds
them through flow (conditioned on prompt_wav's voice/embedding), then hifigan.

Using the SAME real clip as both source and prompt turns this into a self-reconstruction
test: how faithfully flow+hifigan alone can rebuild real speech from its own real tokens,
with llm's text->token guessing entirely out of the picture. Compares our finetuned yo-NG
flow+hifigan (highest checkpoints) against the original pretrained release.
"""
import os
import sys

import torch
import torchaudio

sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice")
sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice3

PRETRAINED_DIR = "/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
WORK_ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/bundle_vc_direct"
OUT_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/inference_vc_direct"
os.makedirs(OUT_DIR, exist_ok=True)

REF_WAV = "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-trial/reference-samples/yoruba_ref.wav"

BASE = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual/individual_yo-NG"
FLOW_HIGHEST = f"{BASE}/flow/epoch_30_whole.pt"
HIFIGAN_HIGHEST = f"{BASE}/hifigan/epoch_84_whole.pt"

VARIANTS = {
    "pretrained": {"flow": f"{PRETRAINED_DIR}/flow.pt", "hift": f"{PRETRAINED_DIR}/hift.pt"},
    "finetuned_yo-NG": {"flow": FLOW_HIGHEST, "hift": HIFIGAN_HIGHEST},
}


def clean_flow(src, dst):
    sd = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({k: v for k, v in sd.items() if torch.is_tensor(v)}, dst)


def clean_hifigan(src, dst):
    sd = torch.load(src, map_location="cpu", weights_only=True)
    if any(k.startswith("generator.") for k in sd.keys()):
        cleaned = {k[len("generator."):]: v for k, v in sd.items() if k.startswith("generator.")}
    else:
        cleaned = {k: v for k, v in sd.items() if torch.is_tensor(v)}
    torch.save(cleaned, dst)


def main():
    for name, ckpts in VARIANTS.items():
        bundle_dir = f"{WORK_ROOT}/{name}"
        os.makedirs(bundle_dir, exist_ok=True)
        for asset in ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "CosyVoice-BlankEN"]:
            dst = f"{bundle_dir}/{asset}"
            if not os.path.exists(dst):
                os.symlink(f"{PRETRAINED_DIR}/{asset}", dst)
        # llm.pt unused by inference_vc but required for CosyVoice3.__init__'s self.model.load()
        dst = f"{bundle_dir}/llm.pt"
        if not os.path.exists(dst):
            os.symlink(f"{PRETRAINED_DIR}/llm.pt", dst)

        print(f"=== [{name}] preparing flow/hift ===", flush=True)
        if ckpts["flow"] == f"{PRETRAINED_DIR}/flow.pt":
            dst = f"{bundle_dir}/flow.pt"
            if not os.path.exists(dst):
                os.symlink(ckpts["flow"], dst)
        else:
            clean_flow(ckpts["flow"], f"{bundle_dir}/flow.pt")
        if ckpts["hift"] == f"{PRETRAINED_DIR}/hift.pt":
            dst = f"{bundle_dir}/hift.pt"
            if not os.path.exists(dst):
                os.symlink(ckpts["hift"], dst)
        else:
            clean_hifigan(ckpts["hift"], f"{bundle_dir}/hift.pt")

        print(f"=== [{name}] loading bundle ===", flush=True)
        model = CosyVoice3(bundle_dir, fp16=False)

        print(f"=== [{name}] real speech -> flow -> hifigan (source == prompt, self-reconstruction) ===", flush=True)
        results = list(model.inference_vc(REF_WAV, REF_WAV, stream=False))
        audio = results[0]["tts_speech"]
        out_path = f"{OUT_DIR}/{name}.wav"
        torchaudio.save(out_path, audio, model.sample_rate)
        dur = audio.shape[1] / model.sample_rate
        peak = audio.abs().max().item()
        print(f"=== [{name}] SAVED: {out_path} ({dur:.2f}s, peak={peak:.4f}) ===", flush=True)
        del model

    print("\nall variants done", flush=True)


if __name__ == "__main__":
    main()
