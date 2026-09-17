"""
Hausa generations for the team: one voice-clone (a specific real reference voice, "clone this
person") and two voice-design variants (different fixed reference voices, used in the "pick a
voice" spirit rather than cloning a particular individual for one request -- CosyVoice3 has no
true voiceless TTS mode, confirmed earlier this session, so this fixed-reference approach is
the real equivalent). All three use ha-NG's best (pre-divergence) llm + original flow (no
finetuned Hausa flow exists) + original vocoder, and the same fresh target text never used
before this session, for a clean comparison.
"""
import os
import sys

import torch
import torchaudio

sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice")
sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice3

PRETRAINED_DIR = "/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
BUNDLE_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/bundle_hausa_drive"
OUT_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/inference_hausa_drive"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_TEXT = "Abu na farko da na ke son faɗa muku shi ne, babu wasu bayanai naku da zan faɗa wa wani."
BEST_LLM = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual/individual_ha-NG/llm/epoch_1_whole.pt"

CASES = {
    "voice_clone": {
        "wav": "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out/hausa/audio/TWB-Voice-1.0 Hausa - hau/354.wav",
        "text": "Anya ku na da masaniyar cewa za ku iya tambaya ta da Hausa ko Turanci ko kuma Kanuri kai har ma da Shuwa?<|endofprompt|>",
    },
    "voice_design_1": {
        "wav": "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out/hausa/audio/Waxal_NLP_Hausa/hau_1521.wav",
        "text": "Idan aka wayi gari aka samu ma'aunin selshiyos ya kai daga 10C zuwa 15C za a samu yanayin sanyi.<|endofprompt|>",
    },
    "voice_design_2": {
        "wav": "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out/hausa/audio/fleurs_ha_ng/10255951444507289726.wav",
        "text": "A farkon lamari, kabilar Byzantine da ke gabashi ta karfafa saka tufafi.<|endofprompt|>",
    },
}


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

    print("=== loading bundle (best llm + original flow + original vocoder) ===", flush=True)
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
