"""
Igbo, "purely the model" -- text only, every prompt/embedding argument at its empty default,
no reference audio at all. Same experiment already run for Yoruba, which crashed in flow's
ODE solver (flow_matching.py: `spks_in[0] = spks` on an empty speaker tensor) -- confirming
CosyVoice3 architecturally requires some real voice conditioning for flow. Running it again
here to confirm this is a real architectural limit and not something specific to the Yoruba
checkpoint.
"""
import os
import sys

import torch
import torchaudio

sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice")
sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice3

PRETRAINED_DIR = "/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
BUNDLE_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/bundle_igbo_no_clone"
OUT_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/inference_igbo_no_clone"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_TEXT = "Olusegun Obasanjo chịrị Naijiria ugboro abụọ, dịka onye agha na onye ọchịchị onye kwuo uche ya.<|endofprompt|>"
BEST_LLM = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual/individual_ig-NG/llm/epoch_1_whole.pt"


def clean(src, dst):
    sd = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({k: v for k, v in sd.items() if torch.is_tensor(v)}, dst)


def main():
    os.makedirs(BUNDLE_DIR, exist_ok=True)
    for asset in ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "CosyVoice-BlankEN", "flow.pt", "hift.pt"]:
        dst = f"{BUNDLE_DIR}/{asset}"
        if not os.path.exists(dst):
            os.symlink(f"{PRETRAINED_DIR}/{asset}", dst)
    clean(BEST_LLM, f"{BUNDLE_DIR}/llm.pt")

    print("=== loading bundle ===", flush=True)
    model = CosyVoice3(BUNDLE_DIR, fp16=False)

    print("=== extracting text tokens only, no reference audio at all ===", flush=True)
    text_token, text_token_len = model.frontend._extract_text_token(TARGET_TEXT)
    print(f"  text tokens: {text_token.shape}", flush=True)

    print("=== calling model.tts() with every prompt/embedding argument at its empty default ===", flush=True)
    try:
        results = list(model.model.tts(text=text_token, text_len=text_token_len, stream=False))
        audio = results[0]["tts_speech"]
        out_path = f"{OUT_DIR}/igbo_no_clone.wav"
        torchaudio.save(out_path, audio, model.sample_rate)
        dur = audio.shape[1] / model.sample_rate
        peak = audio.abs().max().item()
        print(f"=== SUCCESS: {out_path} ({dur:.2f}s, peak={peak:.4f}) ===", flush=True)
    except Exception as e:
        print(f"=== FAILED: {type(e).__name__}: {e} ===", flush=True)
        raise


if __name__ == "__main__":
    main()
