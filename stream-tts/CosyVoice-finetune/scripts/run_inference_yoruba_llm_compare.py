"""
Isolates the llm stage: a real Yoruba voice prompt + its real transcript, and a real target
text, both from yo-NG's own training data, synthesized with our finetuned yo-NG llm vs. the
original pretrained llm. flow and hifigan are the ORIGINAL pretrained weights in both cases --
vocoder finetuning is retired, and holding flow constant too isolates whatever difference is
heard to the llm alone.

An earlier version of this script used an untranscribed reference clip with cross-lingual
mode (no prompt_text) or zero-shot with an empty prompt_text -- both reproducibly crashed
CosyVoice3's llm (mix_ratio-based text/speech interleaving apparently can't handle an empty
prompt_text), identically for both the finetuned and original llm. Using a real prompt+
transcript pair from training data avoids this entirely.
"""
import os
import sys

import torch
import torchaudio

sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice")
sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice3

PRETRAINED_DIR = "/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
WORK_ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/bundle_yoruba_llm_compare"
OUT_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/inference_yoruba_llm_compare"
os.makedirs(OUT_DIR, exist_ok=True)

PROMPT_WAV = "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out/yoruba/audio/fleurs_yo_ng/10615218043355611293.wav"
PROMPT_TEXT = "Mi kò mọ̀ bóyá ó yé ọ tàbí kò yé ọ, sùgbọ́n ọ̀pọ̀ àwọn ẹrù ni ó dé láti ààrin gbùngbùn America wọ orílèèdè yìí lọ́ọ̀fẹ́.<|endofprompt|>"
TARGET_TEXT = "Àwọn aṣàwárí dalábàá pé, bí ètí bá jẹ́ ìrù ọmọ dinosaur, àpẹrẹ rẹ̀ ṣàfihàn ti àgbàlagbà kìn ṣe ti ọmọdé."

FINETUNED_LLM = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual/individual_yo-NG/llm/epoch_46_step_90000.pt"

VARIANTS = {
    "finetuned_llm": FINETUNED_LLM,
    "original_llm": f"{PRETRAINED_DIR}/llm.pt",
}


def clean_llm(src, dst):
    sd = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({k: v for k, v in sd.items() if torch.is_tensor(v)}, dst)


def main():
    for name, llm_ckpt in VARIANTS.items():
        bundle_dir = f"{WORK_ROOT}/{name}"
        os.makedirs(bundle_dir, exist_ok=True)
        for asset in ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "CosyVoice-BlankEN", "flow.pt", "hift.pt"]:
            dst = f"{bundle_dir}/{asset}"
            if not os.path.exists(dst):
                os.symlink(f"{PRETRAINED_DIR}/{asset}", dst)

        print(f"=== [{name}] preparing llm ===", flush=True)
        if llm_ckpt == f"{PRETRAINED_DIR}/llm.pt":
            dst = f"{bundle_dir}/llm.pt"
            if not os.path.exists(dst):
                os.symlink(llm_ckpt, dst)
        else:
            clean_llm(llm_ckpt, f"{bundle_dir}/llm.pt")

        print(f"=== [{name}] loading bundle ===", flush=True)
        model = CosyVoice3(bundle_dir, fp16=False)

        print(f"=== [{name}] zero-shot synthesis (real voice prompt + real transcript, real target text) ===", flush=True)
        last_err = None
        for attempt in range(4):
            try:
                results = list(model.inference_zero_shot(TARGET_TEXT, PROMPT_TEXT, PROMPT_WAV, stream=False))
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
        del model

    print("\nboth variants done", flush=True)


if __name__ == "__main__":
    main()
