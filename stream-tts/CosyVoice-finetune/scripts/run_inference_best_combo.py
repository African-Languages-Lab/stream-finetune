"""
Best-checkpoint (pre-divergence) versions of the llm-isolation test, for both Yoruba and
Hausa. "Best" = the checkpoint matching tensorboard's CV/loss minimum, i.e. before the
CV-loss divergence pattern established earlier this session set in -- as opposed to
"highest" (latest by step count), which the llm-only test showed collapses to near-silent
output on unseen prompt/text pairs for both languages.

Yoruba: best llm + best flow + original vocoder (both finetuned checkpoints exist).
Hausa: best llm + ORIGINAL flow + original vocoder (ha-NG has no trained flow checkpoint at
any point in its history -- only init.pt -- so "best flow" doesn't exist any more than
"highest flow" did; this is the closest real combination available).
"""
import os
import sys

import torch
import torchaudio

sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice")
sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice3

PRETRAINED_DIR = "/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
WORK_ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/bundle_best_combo"
OUT_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/inference_best_combo"
os.makedirs(OUT_DIR, exist_ok=True)

BASE = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual"

CASES = {
    "yoruba": {
        "prompt_wav": "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out/yoruba/audio/fleurs_yo_ng/10344632553186011005.wav",
        "prompt_text": "Àwọn safari ló fẹ́rẹ̀ jẹ́ òǹfà ìrìnafẹ́ jùlọ ní ilẹ̀ adúláwọ̀ àti ohun táwọn àlejò ma ń péwò jù.<|endofprompt|>",
        "target_text": "Àwọn aṣàwárí dalábàá pé, bí ètí bá jẹ́ ìrù ọmọ dinosaur, àpẹrẹ rẹ̀ ṣàfihàn ti àgbàlagbà kìn ṣe ti ọmọdé.",
        "llm": f"{BASE}/individual_yo-NG/llm/epoch_3_whole.pt",
        "flow": f"{BASE}/individual_yo-NG/flow/epoch_18_whole.pt",  # best exists for yo-NG
    },
    "hausa": {
        "prompt_wav": "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out/hausa/audio/TWB-Voice-1.0 Hausa - hau/354.wav",
        "prompt_text": "Anya ku na da masaniyar cewa za ku iya tambaya ta da Hausa ko Turanci ko kuma Kanuri kai har ma da Shuwa?<|endofprompt|>",
        "target_text": "Akwai hanyoyi da yawa da za a bi don kare faruwar laifukan cin zarafi ko fyaɗe a unguwanni.",
        "llm": f"{BASE}/individual_ha-NG/llm/epoch_1_whole.pt",
        "flow": None,  # no trained flow checkpoint exists for ha-NG at any point -- use original
    },
}


def clean(src, dst):
    sd = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({k: v for k, v in sd.items() if torch.is_tensor(v)}, dst)


def main():
    for name, case in CASES.items():
        bundle_dir = f"{WORK_ROOT}/{name}"
        os.makedirs(bundle_dir, exist_ok=True)
        for asset in ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "CosyVoice-BlankEN", "hift.pt"]:
            dst = f"{bundle_dir}/{asset}"
            if not os.path.exists(dst):
                os.symlink(f"{PRETRAINED_DIR}/{asset}", dst)

        print(f"=== [{name}] preparing llm (best) ===", flush=True)
        clean(case["llm"], f"{bundle_dir}/llm.pt")

        print(f"=== [{name}] preparing flow ({'best' if case['flow'] else 'original -- no finetuned flow exists'}) ===", flush=True)
        if case["flow"] is not None:
            clean(case["flow"], f"{bundle_dir}/flow.pt")
        else:
            dst = f"{bundle_dir}/flow.pt"
            if not os.path.exists(dst):
                os.symlink(f"{PRETRAINED_DIR}/flow.pt", dst)

        print(f"=== [{name}] loading bundle ===", flush=True)
        model = CosyVoice3(bundle_dir, fp16=False)

        print(f"=== [{name}] zero-shot synthesis (best llm + {'best' if case['flow'] else 'original'} flow + original vocoder) ===", flush=True)
        last_err = None
        for attempt in range(4):
            try:
                results = list(model.inference_zero_shot(case["target_text"], case["prompt_text"], case["prompt_wav"], stream=False))
                audio = results[0]["tts_speech"]
                out_path = f"{OUT_DIR}/{name}_best.wav"
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

    print("\nall cases done", flush=True)


if __name__ == "__main__":
    main()
