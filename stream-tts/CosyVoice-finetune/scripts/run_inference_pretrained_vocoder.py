"""
Generates audio for many CosyVoice3 regimes using each regime's BEST (lowest CV loss) and
HIGHEST (most trained) llm+flow checkpoints, paired with the ORIGINAL, NEVER-FINE-TUNED
pretrained hift.pt -- not any of our own hifigan checkpoints, all of which are far short of
the step count a from-scratch GAN vocoder needs to sound clean. The pretrained vocoder is
fully converged; this tests whether llm+flow quality alone, rendered through a mature vocoder,
sounds better than anything we can currently produce with our own undertrained hifigan.
"""
import json
import os
import sys

import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice")
sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice3

PRETRAINED_DIR = "/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
KALDI_ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/kaldi_data"
WORK_ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/bundle_pretrained_vocoder"
OUT_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/inference_pretrained_vocoder"

with open("/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/best_highest_result.json") as f:
    CKPTS = json.load(f)


def clean_llm_or_flow(src, dst):
    sd = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({k: v for k, v in sd.items() if torch.is_tensor(v)}, dst)


def pick_reference_and_target(lang):
    text_path = f"{KALDI_ROOT}/individual_{lang}/dev/text"
    wav_path = f"{KALDI_ROOT}/individual_{lang}/dev/wav.scp"
    if not os.path.exists(text_path) or not os.path.exists(wav_path):
        return None, None, None
    texts, wavs = {}, {}
    with open(text_path) as f:
        for line in f:
            p = line.rstrip("\n").split(" ", 1)
            if len(p) == 2:
                texts[p[0]] = p[1]
    with open(wav_path) as f:
        for line in f:
            p = line.rstrip("\n").split(" ", 1)
            if len(p) == 2:
                wavs[p[0]] = p[1]
    candidates = []
    for uid, txt in texts.items():
        if uid not in wavs or not os.path.exists(wavs[uid]) or not txt.strip():
            continue
        try:
            dur = sf.info(wavs[uid]).duration
        except Exception:
            continue
        candidates.append((uid, wavs[uid], txt, dur))
    in_range = [c for c in candidates if 8.0 <= c[3] <= 15.0]
    pool = in_range if len(in_range) >= 2 else [c for c in candidates if c[3] >= 6.0]
    if len(pool) < 2:
        return None, None, None
    _, prompt_wav, prompt_text, _ = pool[0]
    _, _, target_text, _ = pool[1]
    return prompt_wav, prompt_text, target_text


def assemble_and_run(lang, combo, llm_src, flow_src):
    bundle_dir = f"{WORK_ROOT}/{lang}_{combo}"
    os.makedirs(bundle_dir, exist_ok=True)
    for asset in ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "CosyVoice-BlankEN", "hift.pt"]:
        dst = f"{bundle_dir}/{asset}"
        if not os.path.exists(dst):
            os.symlink(f"{PRETRAINED_DIR}/{asset}", dst)
    clean_llm_or_flow(llm_src, f"{bundle_dir}/llm.pt")
    clean_llm_or_flow(flow_src, f"{bundle_dir}/flow.pt")

    prompt_wav, prompt_text, target_text = pick_reference_and_target(lang)
    if prompt_wav is None:
        print(f"  [{lang}/{combo}] SKIP: no usable reference clip", flush=True)
        return

    print(f"  [{lang}/{combo}] loading bundle", flush=True)
    model = CosyVoice3(bundle_dir, fp16=False)
    full_prompt_text = prompt_text + "<|endofprompt|>"
    try:
        results = list(model.inference_zero_shot(target_text, full_prompt_text, prompt_wav, stream=False))
        audio = results[0]["tts_speech"]
        out_path = f"{OUT_DIR}/{lang}_{combo}.wav"
        torchaudio.save(out_path, audio, model.sample_rate)
        dur = audio.shape[1] / model.sample_rate
        peak = audio.abs().max().item()
        print(f"  [{lang}/{combo}] SAVED: {out_path} ({dur:.2f}s, peak={peak:.4f})", flush=True)
    except Exception as e:
        print(f"  [{lang}/{combo}] FAILED: {e}", flush=True)
    del model


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for lang, stages in CKPTS.items():
        print(f"=== {lang} ===", flush=True)
        for combo in ["highest", "best"]:
            llm = stages["llm"][combo]
            flow = stages["flow"][combo]
            if not llm or not flow:
                print(f"  [{lang}/{combo}] SKIP: missing checkpoint", flush=True)
                continue
            assemble_and_run(lang, combo, llm, flow)


if __name__ == "__main__":
    main()
