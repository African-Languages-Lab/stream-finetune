"""
On-demand Hausa generation: recent ft llm + recent ft flow + original vocoder, reusing the
already-built proven bundle (hf_bundles_proven/ha-NG, same checkpoints already pushed to
all-lab/cosyvoice3-individual-ha-NG). Natural (non-Waxal) reference pulled fresh from ha-NG's
own kaldi dev data.
"""
import glob
import json
import os

import soundfile as sf
import sys
import torchaudio

sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice")
sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice3

BUNDLE_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/hf_bundles_proven/ha-NG"
KALDI_ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/kaldi_data"
OUT_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/inference_hausa_stream_now"
os.makedirs(OUT_DIR, exist_ok=True)


def pick_prompt_and_target(code):
    kaldi_dir = f"{KALDI_ROOT}/individual_{code}"
    wavscp, text = {}, {}
    with open(f"{kaldi_dir}/dev/wav.scp") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                wavscp[parts[0]] = parts[1]
    with open(f"{kaldi_dir}/dev/text") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                text[parts[0]] = parts[1]

    candidates = []
    for uid, path in wavscp.items():
        if uid not in text or not os.path.exists(path) or "Waxal_NLP" in path:
            continue
        try:
            dur = sf.info(path).duration
        except Exception:
            continue
        candidates.append((uid, path, dur, text[uid]))

    prompt = next(((u, p, d, t) for u, p, d, t in candidates if 7.0 <= d <= 15.0 and len(t) >= 15), None)
    target = next(((u, p, d, t) for u, p, d, t in candidates if prompt and u != prompt[0] and 60 <= len(t) <= 180), None)
    return {
        "prompt_wav": prompt[1],
        "prompt_text": prompt[3] + "<|endofprompt|>",
        "target_text": target[3],
    }


def main():
    pt = pick_prompt_and_target("ha-NG")
    print("prompt:", pt["prompt_wav"], flush=True)
    print("prompt_text:", pt["prompt_text"], flush=True)
    print("target_text:", pt["target_text"], flush=True)

    model = CosyVoice3(BUNDLE_DIR, fp16=False)
    last_err = None
    for attempt in range(4):
        try:
            results = list(model.inference_zero_shot(pt["target_text"], pt["prompt_text"], pt["prompt_wav"], stream=False))
            audio = results[0]["tts_speech"]
            out_path = f"{OUT_DIR}/ha-NG_stream_now.wav"
            torchaudio.save(out_path, audio, model.sample_rate)
            dur = audio.shape[1] / model.sample_rate
            peak = audio.abs().max().item()
            print(f"SAVED: {out_path} ({dur:.2f}s, peak={peak:.4f})", flush=True)
            with open(f"{OUT_DIR}/manifest.json", "w") as f:
                json.dump({"ha-NG": {**pt, "duration": dur, "peak": peak, "llm": "recent", "flow": "recent", "vocoder": "original"}}, f, indent=2, ensure_ascii=False)
            return
        except RuntimeError as e:
            last_err = e
            print(f"attempt {attempt + 1} failed ({e}), retrying", flush=True)
    print(f"FAILED after retries: {last_err}", flush=True)


if __name__ == "__main__":
    main()
