"""
Yoruba synthesis with our finetuned llm (highest) + our finetuned flow (highest) + the
ORIGINAL pretrained vocoder (hifigan finetuning is retired). Same real voice prompt +
transcript and real target text used in the llm-only comparison, so this is directly
comparable to that result.
"""
import os
import sys

import torch
import torchaudio

sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice")
sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice3

PRETRAINED_DIR = "/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
BUNDLE_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/bundle_yoruba_llm_flow_ft"
OUT_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/inference_yoruba_llm_flow_ft"
os.makedirs(OUT_DIR, exist_ok=True)

PROMPT_WAV = "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out/yoruba/audio/fleurs_yo_ng/10615218043355611293.wav"
PROMPT_TEXT = "Mi kò mọ̀ bóyá ó yé ọ tàbí kò yé ọ, sùgbọ́n ọ̀pọ̀ àwọn ẹrù ni ó dé láti ààrin gbùngbùn America wọ orílèèdè yìí lọ́ọ̀fẹ́.<|endofprompt|>"
TARGET_TEXT = "Àwọn aṣàwárí dalábàá pé, bí ètí bá jẹ́ ìrù ọmọ dinosaur, àpẹrẹ rẹ̀ ṣàfihàn ti àgbàlagbà kìn ṣe ti ọmọdé."

FINETUNED_LLM = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual/individual_yo-NG/llm/epoch_46_step_90000.pt"
FINETUNED_FLOW = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual/individual_yo-NG/flow/epoch_30_whole.pt"


def clean(src, dst):
    sd = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({k: v for k, v in sd.items() if torch.is_tensor(v)}, dst)


def main():
    os.makedirs(BUNDLE_DIR, exist_ok=True)
    for asset in ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "CosyVoice-BlankEN", "hift.pt"]:
        dst = f"{BUNDLE_DIR}/{asset}"
        if not os.path.exists(dst):
            os.symlink(f"{PRETRAINED_DIR}/{asset}", dst)

    print("=== cleaning llm + flow ===", flush=True)
    clean(FINETUNED_LLM, f"{BUNDLE_DIR}/llm.pt")
    clean(FINETUNED_FLOW, f"{BUNDLE_DIR}/flow.pt")

    print("=== loading bundle ===", flush=True)
    model = CosyVoice3(BUNDLE_DIR, fp16=False)

    print("=== zero-shot synthesis (finetuned llm + finetuned flow + original vocoder) ===", flush=True)
    last_err = None
    for attempt in range(4):
        try:
            results = list(model.inference_zero_shot(TARGET_TEXT, PROMPT_TEXT, PROMPT_WAV, stream=False))
            audio = results[0]["tts_speech"]
            out_path = f"{OUT_DIR}/yoruba_ft_llm_ft_flow.wav"
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
