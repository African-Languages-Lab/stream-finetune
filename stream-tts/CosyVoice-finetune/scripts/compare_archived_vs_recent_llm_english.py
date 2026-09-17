"""
English archived-ft-llm vs recent-ft-llm comparison, en-UG only.

Of the four English accent regimes (en-GH, en-NG, en-UG, en-ZA), only en-UG was actually
trained to completion in either checkpoint root -- en-GH/en-NG/en-ZA were stopped with zero
checkpoints (superseded by the "combined" all-41-languages regime, per plan). The combined
regime pools every language rather than being English-accent-specific, and as of this script's
writing it has just been reset after an Aug 22 tokenizer update and has no trained llm
checkpoint yet (init.pt only, in both the current and archived checkpoint roots) -- so it
can't stand in for "all the accents" either. This script therefore covers en-UG alone; rerun
once en-GH/en-NG/en-ZA or combined actually produce checkpoints.

Same treatment as the 5-language comparison: archived_llm vs recent_llm, both built with the
SAME recent/current ft flow and the SAME original pretrained vocoder, and the SAME
prompt+target reference pair (from en-UG's own kaldi dev data) for both variants.
"""
import glob
import json
import os
import sys
import traceback

import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice")
sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice3

PRETRAINED_DIR = "/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
CURRENT_ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual"
ARCHIVE_ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints_archived_20260818/individual"
KALDI_ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/kaldi_data"
WORK_ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/bundle_archived_vs_recent_english"
OUT_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/inference_archived_vs_recent_english"
MANIFEST_PATH = f"{OUT_DIR}/manifest.json"
os.makedirs(OUT_DIR, exist_ok=True)

CODE = "en-UG"


def highest_checkpoint(stage_dir):
    files = [p for p in glob.glob(f"{stage_dir}/*.pt") if os.path.basename(p) != "init.pt"]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def best_checkpoint(stage_dir):
    tb_dir = stage_dir.replace("/llm", "/tensorboard/llm").replace("/flow", "/tensorboard/flow")
    try:
        from tensorboard.backend.event_processing import event_accumulator
        ea = event_accumulator.EventAccumulator(tb_dir, size_guidance={"scalars": 0})
        ea.Reload()
        events = ea.Scalars("CV/loss")
        best_event = min(events, key=lambda e: e.value)
        files = [p for p in glob.glob(f"{stage_dir}/*.pt") if os.path.basename(p) != "init.pt"]
        if not files:
            return None
        return min(files, key=lambda p: abs(os.path.getmtime(p) - best_event.wall_time))
    except Exception:
        return None


def pick_prompt_and_target(code):
    kaldi_dir = f"{KALDI_ROOT}/individual_{code}"
    wavscp_path = f"{kaldi_dir}/dev/wav.scp"
    text_path = f"{kaldi_dir}/dev/text"
    wavscp, text = {}, {}
    with open(wavscp_path) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                wavscp[parts[0]] = parts[1]
    with open(text_path) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                text[parts[0]] = parts[1]

    candidates = []
    for uid, path in wavscp.items():
        if uid not in text or not os.path.exists(path):
            continue
        try:
            dur = sf.info(path).duration
        except Exception:
            continue
        candidates.append((uid, path, dur, text[uid]))

    prompt = next(((u, p, d, t) for u, p, d, t in candidates if 7.0 <= d <= 15.0 and len(t) >= 15), None)
    if prompt is None:
        candidates_sorted = sorted(candidates, key=lambda c: c[2], reverse=True)
        prompt = candidates_sorted[0] if candidates_sorted else None

    target = next(
        ((u, p, d, t) for u, p, d, t in candidates if u != prompt[0] and 60 <= len(t) <= 180),
        None,
    )
    if target is None:
        others = [c for c in candidates if c[0] != prompt[0]]
        target = max(others, key=lambda c: len(c[3])) if others else None

    return {
        "prompt_wav": prompt[1],
        "prompt_text": prompt[3] + "<|endofprompt|>",
        "target_text": target[3],
    }


def clean(src, dst):
    sd = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({k: v for k, v in sd.items() if torch.is_tensor(v)}, dst)


def build_bundle(bundle_dir, llm_ckpt, flow_ckpt):
    os.makedirs(bundle_dir, exist_ok=True)
    assets = ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "CosyVoice-BlankEN", "hift.pt"]
    if flow_ckpt is None:
        assets.append("flow.pt")
    for asset in assets:
        dst = f"{bundle_dir}/{asset}"
        if not os.path.exists(dst):
            os.symlink(f"{PRETRAINED_DIR}/{asset}", dst)
    clean(llm_ckpt, f"{bundle_dir}/llm.pt")
    if flow_ckpt is not None:
        clean(flow_ckpt, f"{bundle_dir}/flow.pt")


def synthesize(bundle_dir, prompt_wav, prompt_text, target_text, out_path):
    model = CosyVoice3(bundle_dir, fp16=False)
    last_err = None
    for attempt in range(4):
        try:
            results = list(model.inference_zero_shot(target_text, prompt_text, prompt_wav, stream=False))
            audio = results[0]["tts_speech"]
            torchaudio.save(out_path, audio, model.sample_rate)
            dur = audio.shape[1] / model.sample_rate
            peak = audio.abs().max().item()
            del model
            return {"ok": True, "duration": dur, "peak": peak}
        except RuntimeError as e:
            last_err = str(e)
    del model
    return {"ok": False, "error": last_err}


def main():
    manifest = {}
    print(f"=== English comparison: {CODE} only (see module docstring for why) ===", flush=True)

    recent_llm = best_checkpoint(f"{CURRENT_ROOT}/individual_{CODE}/llm") or highest_checkpoint(f"{CURRENT_ROOT}/individual_{CODE}/llm")
    archived_llm = best_checkpoint(f"{ARCHIVE_ROOT}/individual_{CODE}/llm") or highest_checkpoint(f"{ARCHIVE_ROOT}/individual_{CODE}/llm")
    recent_flow = best_checkpoint(f"{CURRENT_ROOT}/individual_{CODE}/flow") or highest_checkpoint(f"{CURRENT_ROOT}/individual_{CODE}/flow")

    print(f"  recent_llm={recent_llm}", flush=True)
    print(f"  archived_llm={archived_llm}", flush=True)
    print(f"  recent_flow (used for both variants)={recent_flow}", flush=True)

    pt = pick_prompt_and_target(CODE)
    print(f"  prompt={pt['prompt_wav']}", flush=True)
    print(f"  target_text={pt['target_text']}", flush=True)

    manifest[CODE] = {
        "status": "attempted",
        "recent_llm_checkpoint": recent_llm,
        "archived_llm_checkpoint": archived_llm,
        "recent_flow_checkpoint": recent_flow,
        "prompt_wav": pt["prompt_wav"],
        "prompt_text": pt["prompt_text"],
        "target_text": pt["target_text"],
        "variants": {},
    }

    for variant_name, llm_ckpt in [("archived_llm", archived_llm), ("recent_llm", recent_llm)]:
        bundle_dir = f"{WORK_ROOT}/{CODE}_{variant_name}"
        out_path = f"{OUT_DIR}/{CODE}_{variant_name}.wav"
        try:
            print(f"=== [{variant_name}] building bundle ===", flush=True)
            build_bundle(bundle_dir, llm_ckpt, recent_flow)
            print(f"=== [{variant_name}] synthesizing ===", flush=True)
            result = synthesize(bundle_dir, pt["prompt_wav"], pt["prompt_text"], pt["target_text"], out_path)
            if result["ok"]:
                print(f"=== [{variant_name}] SAVED: {out_path} ({result['duration']:.2f}s, peak={result['peak']:.4f}) ===", flush=True)
            else:
                print(f"=== [{variant_name}] FAILED: {result['error']} ===", flush=True)
            manifest[CODE]["variants"][variant_name] = result
        except Exception as e:
            print(f"=== [{variant_name}] EXCEPTION: {e} ===", flush=True)
            traceback.print_exc()
            manifest[CODE]["variants"][variant_name] = {"ok": False, "error": str(e)}

        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n=== done ===", flush=True)


if __name__ == "__main__":
    main()
