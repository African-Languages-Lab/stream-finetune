"""
"Just TTS, not cloning" -- text only, no reference voice at all. CosyVoice3-0.5B ships with
no spk2info.pt (confirmed: file doesn't exist in the release dir), so there's no baked-in
default speaker bank -- inference_sft() would crash outright. The only real option is calling
model.tts() directly with every prompt/embedding argument left at its zero-length default.

The llm side explicitly supports this (cosyvoice/llm/llm.py: `if prompt_speech_token_len != 0
... else: torch.zeros(...)`), so llm generation from text alone is a real, intentional code
path, not a hack. Whether flow tolerates a fully empty prompt too is untested -- that's what
this script actually checks, by just running it and reporting exactly what happens.

Uses Yoruba's best (pre-divergence, healthy) llm checkpoint + original flow + original
vocoder, same target text as the cloning test for direct comparison.
"""
import os
import sys

import torch
import torchaudio

sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice")
sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice3

PRETRAINED_DIR = "/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
BUNDLE_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/bundle_no_clone"
OUT_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/inference_no_clone"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_TEXT = "Àwọn aṣàwárí dalábàá pé, bí ètí bá jẹ́ ìrù ọmọ dinosaur, àpẹrẹ rẹ̀ ṣàfihàn ti àgbàlagbà kìn ṣe ti ọmọdé.<|endofprompt|>"
FINETUNED_LLM = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual/individual_yo-NG/llm/epoch_3_whole.pt"


def clean(src, dst):
    sd = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({k: v for k, v in sd.items() if torch.is_tensor(v)}, dst)


def main():
    os.makedirs(BUNDLE_DIR, exist_ok=True)
    for asset in ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "CosyVoice-BlankEN", "flow.pt", "hift.pt"]:
        dst = f"{BUNDLE_DIR}/{asset}"
        if not os.path.exists(dst):
            os.symlink(f"{PRETRAINED_DIR}/{asset}", dst)
    clean(FINETUNED_LLM, f"{BUNDLE_DIR}/llm.pt")

    print("=== loading bundle ===", flush=True)
    model = CosyVoice3(BUNDLE_DIR, fp16=False)

    print("=== extracting text tokens only, no reference audio at all ===", flush=True)
    text_token, text_token_len = model.frontend._extract_text_token(TARGET_TEXT)
    print(f"  text tokens: {text_token.shape}", flush=True)

    print("=== calling model.tts() with every prompt/embedding argument at its empty default ===", flush=True)
    try:
        results = list(model.model.tts(text=text_token, text_len=text_token_len, stream=False))
        audio = results[0]["tts_speech"]
        out_path = f"{OUT_DIR}/yoruba_no_clone.wav"
        torchaudio.save(out_path, audio, model.sample_rate)
        dur = audio.shape[1] / model.sample_rate
        peak = audio.abs().max().item()
        print(f"=== SUCCESS: {out_path} ({dur:.2f}s, peak={peak:.4f}) ===", flush=True)
    except Exception as e:
        print(f"=== FAILED: {type(e).__name__}: {e} ===", flush=True)
        raise


if __name__ == "__main__":
    main()
