"""
Igbo audio: highest llm (new restart), highest flow (old archive -- new restart hasn't
reached flow yet, but the archive has 188 real flow checkpoints, far more mature than what
crashed for bem-ZM earlier), pretrained (never fine-tuned) hift.pt.
"""
import os
import sys

import torch
import torchaudio

sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice")
sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice3

PRETRAINED_DIR = "/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
BUNDLE_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/bundle_igbo_highest"
OUT_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/inference_igbo_highest"

LLM_SRC = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual/individual_ig-NG/llm/epoch_45_whole.pt"
FLOW_SRC = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints_archived_20260818/individual/individual_ig-NG/flow/epoch_35_step_306000.pt"

PROMPT_WAV = "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out/igbo/audio/Waxal_NLP_Igbo/ibo_6.wav"
PROMPT_TEXT = "ato ibe ano, n'ime okirikiri"
TARGET_TEXT = "Ndị uwe ojii dosiri akara okporo ụzọ na-acha pinki na edo edo n'okporo ụzọ awara awara. Akara a ka ha ji egosi ndị na-akwọ ụgbọala ụzọ ha kwesịrị isi."


def clean(src, dst):
    sd = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({k: v for k, v in sd.items() if torch.is_tensor(v)}, dst)


def main():
    os.makedirs(BUNDLE_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    for asset in ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "CosyVoice-BlankEN", "hift.pt"]:
        dst = f"{BUNDLE_DIR}/{asset}"
        if not os.path.exists(dst):
            os.symlink(f"{PRETRAINED_DIR}/{asset}", dst)
    print("=== cleaning llm ===", flush=True)
    clean(LLM_SRC, f"{BUNDLE_DIR}/llm.pt")
    print("=== cleaning flow ===", flush=True)
    clean(FLOW_SRC, f"{BUNDLE_DIR}/flow.pt")

    print(f"=== loading bundle from {BUNDLE_DIR} ===", flush=True)
    model = CosyVoice3(BUNDLE_DIR, fp16=False)
    full_prompt_text = PROMPT_TEXT + "<|endofprompt|>"
    print(f"prompt_text={full_prompt_text!r}", flush=True)
    print(f"target_text={TARGET_TEXT!r}", flush=True)
    results = list(model.inference_zero_shot(TARGET_TEXT, full_prompt_text, PROMPT_WAV, stream=False))
    audio = results[0]["tts_speech"]
    out_path = f"{OUT_DIR}/ig-NG_highest.wav"
    torchaudio.save(out_path, audio, model.sample_rate)
    dur = audio.shape[1] / model.sample_rate
    peak = audio.abs().max().item()
    print(f"=== SAVED: {out_path} ({dur:.2f}s, peak={peak:.4f}) ===", flush=True)


if __name__ == "__main__":
    main()
