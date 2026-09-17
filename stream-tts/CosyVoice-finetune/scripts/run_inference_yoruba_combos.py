"""
Generates Yoruba audio from two checkpoint combinations for yo-NG:
  BEST  = lowest-CV-loss llm/flow checkpoints + highest-step hifigan (no CV-based "best" for
          hifigan since early-stop was disabled for it -- more steps is the only signal we trust)
  HIGHEST = highest-step (most-trained) checkpoint for every stage, regardless of CV trend

Both combos share the same real prompt voice + real, different target transcript from yo-NG's
own training data.
"""
import os
import sys

import torch
import torchaudio

sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice")
sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice3

PRETRAINED_DIR = "/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
WORK_ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/bundle_yoruba_combos"
OUT_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/inference_yoruba_combos"

BASE = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual/individual_yo-NG"

COMBOS = {
    "best": {
        "llm": f"{BASE}/llm/epoch_3_whole.pt",
        "flow": f"{BASE}/flow/epoch_18_whole.pt",
        "hifigan": f"{BASE}/hifigan/epoch_28_whole.pt",
    },
    "highest": {
        "llm": f"{BASE}/llm/epoch_46_step_90000.pt",
        "flow": f"{BASE}/flow/epoch_30_whole.pt",
        "hifigan": f"{BASE}/hifigan/epoch_28_whole.pt",
    },
}

PROMPT_WAV = "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out/yoruba/audio/Waxal_NLP_Yoruba/yor_1356.wav"
PROMPT_TEXT = "Bólù Wà Nínú Àwọ̀n"
TARGET_TEXT = "Ní orílẹ̀èdè Japan, àwọn olórí nìkan ni ó má ń sayẹyẹ èso ṣẹ́ẹ́ẹ́rì fúnra ara wọn àtàwọn èèkàn nídí iṣẹ́ ọba mìíran nínú ìgbìmọ̀ ọba mìíràn."


def clean_llm_or_flow(src, dst):
    sd = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({k: v for k, v in sd.items() if torch.is_tensor(v)}, dst)


def clean_hifigan(src, dst):
    sd = torch.load(src, map_location="cpu", weights_only=True)
    if any(k.startswith("generator.") for k in sd.keys()):
        cleaned = {k[len("generator."):]: v for k, v in sd.items() if k.startswith("generator.")}
    else:
        cleaned = {k: v for k, v in sd.items() if torch.is_tensor(v)}
    torch.save(cleaned, dst)


def run_combo(name, sources):
    bundle_dir = f"{WORK_ROOT}/{name}"
    os.makedirs(bundle_dir, exist_ok=True)
    for asset in ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "CosyVoice-BlankEN"]:
        dst = f"{bundle_dir}/{asset}"
        if not os.path.exists(dst):
            os.symlink(f"{PRETRAINED_DIR}/{asset}", dst)
    print(f"=== [{name}] cleaning checkpoints ===", flush=True)
    clean_llm_or_flow(sources["llm"], f"{bundle_dir}/llm.pt")
    clean_llm_or_flow(sources["flow"], f"{bundle_dir}/flow.pt")
    clean_hifigan(sources["hifigan"], f"{bundle_dir}/hift.pt")

    print(f"=== [{name}] loading bundle ===", flush=True)
    model = CosyVoice3(bundle_dir, fp16=False)
    full_prompt_text = PROMPT_TEXT + "<|endofprompt|>"
    try:
        results = list(model.inference_zero_shot(TARGET_TEXT, full_prompt_text, PROMPT_WAV, stream=False))
        audio = results[0]["tts_speech"]
        out_path = f"{OUT_DIR}/yo-NG_{name}.wav"
        torchaudio.save(out_path, audio, model.sample_rate)
        dur = audio.shape[1] / model.sample_rate
        peak = audio.abs().max().item()
        print(f"=== [{name}] SAVED: {out_path} ({dur:.2f}s, peak={peak:.4f}) ===", flush=True)
    except Exception as e:
        print(f"=== [{name}] FAILED: {e} ===", flush=True)
    del model


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, sources in COMBOS.items():
        if only in (None, name):
            run_combo(name, sources)
    print("=== all combos done ===", flush=True)
