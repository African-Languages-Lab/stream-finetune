"""
"TTS without voice cloning," realized the only way CosyVoice3's architecture actually
supports: flow has no code path for zero reference conditioning (confirmed by testing it
directly -- it hard-crashes in flow_matching.py's ODE solver on an empty prompt), so true
voice-less generation isn't possible. The practical equivalent is a FIXED "house voice": one
reference clip, reused as a constant default so every call is just text-in, speech-out with
no cloning decision per request. Same fixed prompt as the current best-combo Yoruba result;
brand new target text never used in this session before, to demonstrate it generalizes.
"""
import os
import sys

import torch
import torchaudio

sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice")
sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice3

PRETRAINED_DIR = "/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
BUNDLE_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/bundle_house_voice"
OUT_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/inference_house_voice"
os.makedirs(OUT_DIR, exist_ok=True)

# fixed "house voice" -- the same clip for every call, never swapped per-request
HOUSE_VOICE_WAV = "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out/yoruba/audio/fleurs_yo_ng/10344632553186011005.wav"
HOUSE_VOICE_TEXT = "Àwọn safari ló fẹ́rẹ̀ jẹ́ òǹfà ìrìnafẹ́ jùlọ ní ilẹ̀ adúláwọ̀ àti ohun táwọn àlejò ma ń péwò jù.<|endofprompt|>"

TARGET_TEXT = "Ọjọ́ karùndínlógún oṣù kìn-ní ni ọjọ́ tí orílẹ̀-èdè Nàìjíríà máa ń ṣe ìrántí àyájọ́ ọjọ́ ìrántí àwọn ọmọ ológun tí wọ́n ti sùn lójú ìjà."

FINETUNED_LLM = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual/individual_yo-NG/llm/epoch_3_whole.pt"
FINETUNED_FLOW = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual/individual_yo-NG/flow/epoch_18_whole.pt"


def clean(src, dst):
    sd = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({k: v for k, v in sd.items() if torch.is_tensor(v)}, dst)


def main():
    os.makedirs(BUNDLE_DIR, exist_ok=True)
    for asset in ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "CosyVoice-BlankEN", "hift.pt"]:
        dst = f"{BUNDLE_DIR}/{asset}"
        if not os.path.exists(dst):
            os.symlink(f"{PRETRAINED_DIR}/{asset}", dst)
    clean(FINETUNED_LLM, f"{BUNDLE_DIR}/llm.pt")
    clean(FINETUNED_FLOW, f"{BUNDLE_DIR}/flow.pt")

    print("=== loading bundle (best llm + best flow + original vocoder, fixed house voice) ===", flush=True)
    model = CosyVoice3(BUNDLE_DIR, fp16=False)

    print("=== zero-shot synthesis, house voice + brand new target text ===", flush=True)
    last_err = None
    for attempt in range(4):
        try:
            results = list(model.inference_zero_shot(TARGET_TEXT, HOUSE_VOICE_TEXT, HOUSE_VOICE_WAV, stream=False))
            audio = results[0]["tts_speech"]
            out_path = f"{OUT_DIR}/yoruba_house_voice.wav"
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
