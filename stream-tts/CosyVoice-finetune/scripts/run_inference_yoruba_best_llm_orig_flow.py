"""
Yoruba with best (pre-divergence) llm + ORIGINAL flow + original vocoder -- isolates whether
our finetuned flow is actually helping over the pretrained one, using the same treatment
Hausa already got by necessity (no finetuned Hausa flow exists). Same prompt/target as the
current best-combo Yoruba test, only flow differs.
"""
import os
import sys

import torch
import torchaudio

sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice")
sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice3

PRETRAINED_DIR = "/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
BUNDLE_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/bundle_yoruba_best_llm_orig_flow"
OUT_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/inference_yoruba_best_llm_orig_flow"
os.makedirs(OUT_DIR, exist_ok=True)

PROMPT_WAV = "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out/yoruba/audio/fleurs_yo_ng/10344632553186011005.wav"
PROMPT_TEXT = "Àwọn safari ló fẹ́rẹ̀ jẹ́ òǹfà ìrìnafẹ́ jùlọ ní ilẹ̀ adúláwọ̀ àti ohun táwọn àlejò ma ń péwò jù.<|endofprompt|>"
TARGET_TEXT = "Àwọn aṣàwárí dalábàá pé, bí ètí bá jẹ́ ìrù ọmọ dinosaur, àpẹrẹ rẹ̀ ṣàfihàn ti àgbàlagbà kìn ṣe ti ọmọdé."

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

    print("=== loading bundle (best llm + original flow + original vocoder) ===", flush=True)
    model = CosyVoice3(BUNDLE_DIR, fp16=False)

    print("=== zero-shot synthesis ===", flush=True)
    last_err = None
    for attempt in range(4):
        try:
            results = list(model.inference_zero_shot(TARGET_TEXT, PROMPT_TEXT, PROMPT_WAV, stream=False))
            audio = results[0]["tts_speech"]
            out_path = f"{OUT_DIR}/yoruba_best_llm_orig_flow.wav"
            torchaudio.save(out_path, audio, model.sample_rate)
            dur = audio.shape[1] / model.sample_rate
            peak = audio.abs().max().item()
            print(f"=== SAVED: {out_path} ({dur:.2f}s, peak={peak:.4f}) ===", flush=True)
            last_err = None
            break
        except RuntimeError as e:
            last_err = e
            print(f"=== attempt {attempt + 1} failed ({e}), retrying ===", flush=True)
    if last_err is not None:
        print(f"=== FAILED after retries: {last_err} ===", flush=True)


if __name__ == "__main__":
    main()
