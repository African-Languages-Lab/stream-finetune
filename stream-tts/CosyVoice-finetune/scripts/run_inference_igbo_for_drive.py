"""
Igbo generations for the team: one voice-clone and two voice-design variants, all using
ig-NG's best (pre-divergence) llm + best (CV-loss-minimum) flow + original vocoder. Unlike
Hausa, Igbo genuinely has a finetuned flow checkpoint (from the archived run), so this is the
real ft-llm + ft-flow combo, not a forced original-flow fallback.
"""
import os
import sys

import torch
import torchaudio

sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice")
sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice3

PRETRAINED_DIR = "/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
BUNDLE_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/bundle_igbo_drive"
OUT_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/inference_igbo_drive"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_TEXT = "Ndị mmadụ ka na-agbaso usoro ọzọ iji chekwaba ahụ ike ha n'oge ọrịa na-agbasa."

BEST_LLM = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual/individual_ig-NG/llm/epoch_2_whole.pt"
BEST_FLOW = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints_archived_20260818/individual/individual_ig-NG/flow/epoch_32_step_276000.pt"

CASES = {
    "voice_clone": {
        "wav": "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out/igbo/audio/Waxal_NLP_Igbo/ibo_1436.wav",
        "text": "Onye nkụzi ụmụaka ahụ sere otu okirikiri na-acha anụnụ anụnụ n'etiti rombusu abụọ na-acha edo edo.<|endofprompt|>",
    },
    "voice_design_1": {
        "wav": "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out/igbo/audio/Waxal_NLP_Igbo/ibo_767.wav",
        "text": "Okpomọkụ dị elu maka taa ga-abụ selsịọsụ iri atọ na otu mana obere mmiri ga-ezo ma ọ na eweta aka n'elekere mbụ nke ụtụtụ.<|endofprompt|>",
    },
    "voice_design_2": {
        "wav": "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out/igbo/audio/fleurs_ig_ng/10052828106951583547.wav",
        "text": "Akara ya maka mgbututu n'ụbọchị Tuzde, mana azọpụtara ya ka ụlọ ikpe kpegharịrị ikpe na mberede.<|endofprompt|>",
    },
}


def clean(src, dst):
    sd = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({k: v for k, v in sd.items() if torch.is_tensor(v)}, dst)


def main():
    os.makedirs(BUNDLE_DIR, exist_ok=True)
    for asset in ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "CosyVoice-BlankEN", "hift.pt"]:
        dst = f"{BUNDLE_DIR}/{asset}"
        if not os.path.exists(dst):
            os.symlink(f"{PRETRAINED_DIR}/{asset}", dst)
    clean(BEST_LLM, f"{BUNDLE_DIR}/llm.pt")
    clean(BEST_FLOW, f"{BUNDLE_DIR}/flow.pt")

    print("=== loading bundle (ft llm + ft flow + original vocoder) ===", flush=True)
    model = CosyVoice3(BUNDLE_DIR, fp16=False)

    for name, case in CASES.items():
        print(f"=== [{name}] zero-shot synthesis ===", flush=True)
        last_err = None
        for attempt in range(4):
            try:
                results = list(model.inference_zero_shot(TARGET_TEXT, case["text"], case["wav"], stream=False))
                audio = results[0]["tts_speech"]
                out_path = f"{OUT_DIR}/{name}.wav"
                torchaudio.save(out_path, audio, model.sample_rate)
                dur = audio.shape[1] / model.sample_rate
                peak = audio.abs().max().item()
                print(f"=== [{name}] SAVED: {out_path} ({dur:.2f}s, peak={peak:.4f}) ===", flush=True)
                last_err = None
                break
            except RuntimeError as e:
                last_err = e
                print(f"=== [{name}] attempt {attempt + 1} failed ({e}), retrying ===", flush=True)
        if last_err is not None:
            print(f"=== [{name}] FAILED after retries: {last_err} ===", flush=True)

    print("\nall cases done", flush=True)


if __name__ == "__main__":
    main()
