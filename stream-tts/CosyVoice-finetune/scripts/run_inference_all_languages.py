"""
For every individual regime with a usable llm checkpoint: generate with best llm + best flow
(if any flow exists, current restart preferred over the archived run) + original vocoder,
and best llm + original flow + original vocoder. Both use a real prompt clip + real transcript
and a real, different target text, both pulled automatically from that language's own kaldi
dev data. Original vocoder throughout -- hifigan finetuning is retired.

"Best" = tensorboard CV/loss minimum's wall_time matched to the closest-mtime checkpoint file;
falls back to "highest" (latest by mtime, excluding init.pt) if tensorboard data is missing or
unreadable for that stage. This mirrors the manual selection done earlier this session for
Yoruba/Hausa/Igbo, now automated across all regimes.
"""
import glob
import json
import os
import re
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
WORK_ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/bundle_all_languages"
OUT_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/inference_all_languages"
MANIFEST_PATH = f"{OUT_DIR}/manifest.json"
os.makedirs(OUT_DIR, exist_ok=True)


def list_regime_codes():
    codes = []
    for d in sorted(glob.glob(f"{CURRENT_ROOT}/individual_*")):
        codes.append(os.path.basename(d)[len("individual_"):])
    return codes


def epoch_key(p):
    m = re.search(r"epoch_(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else -1


def highest_checkpoint(stage_dir):
    files = [p for p in glob.glob(f"{stage_dir}/*.pt") if os.path.basename(p) != "init.pt"]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def best_checkpoint(stage_dir):
    """CV/loss minimum's wall_time matched to closest-mtime checkpoint; None if no tensorboard data."""
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


def resolve_llm(code):
    stage_dir = f"{CURRENT_ROOT}/individual_{code}/llm"
    ckpt = best_checkpoint(stage_dir) or highest_checkpoint(stage_dir)
    return ckpt


def resolve_flow(code):
    """Prefer current restart's flow; fall back to archive. Returns None if neither has one."""
    for root in (CURRENT_ROOT, ARCHIVE_ROOT):
        stage_dir = f"{root}/individual_{code}/flow"
        ckpt = best_checkpoint(stage_dir) or highest_checkpoint(stage_dir)
        if ckpt is not None:
            return ckpt
    return None


def pick_prompt_and_target(code):
    kaldi_dir = f"{KALDI_ROOT}/individual_{code}"
    wavscp_path = f"{kaldi_dir}/dev/wav.scp"
    text_path = f"{kaldi_dir}/dev/text"
    if not (os.path.exists(wavscp_path) and os.path.exists(text_path)):
        return None
    wavscp = {}
    with open(wavscp_path) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                wavscp[parts[0]] = parts[1]
    text = {}
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
    if prompt is None:
        return None

    target = next(
        ((u, p, d, t) for u, p, d, t in candidates if u != prompt[0] and 60 <= len(t) <= 180),
        None,
    )
    if target is None:
        others = [c for c in candidates if c[0] != prompt[0]]
        target = max(others, key=lambda c: len(c[3])) if others else None
    if target is None:
        return None

    return {
        "prompt_wav": prompt[1],
        "prompt_text": prompt[3] + "<|endofprompt|>",
        "target_text": target[3],
    }


def clean(src, dst):
    sd = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({k: v for k, v in sd.items() if torch.is_tensor(v)}, dst)


def clean_flow(src, dst):
    sd = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({k: v for k, v in sd.items() if torch.is_tensor(v)}, dst)


def build_bundle(bundle_dir, llm_ckpt, flow_ckpt):
    os.makedirs(bundle_dir, exist_ok=True)
    for asset in ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "CosyVoice-BlankEN", "hift.pt"]:
        dst = f"{bundle_dir}/{asset}"
        if not os.path.exists(dst):
            os.symlink(f"{PRETRAINED_DIR}/{asset}", dst)
    clean(llm_ckpt, f"{bundle_dir}/llm.pt")
    if flow_ckpt is not None:
        clean_flow(flow_ckpt, f"{bundle_dir}/flow.pt")
    else:
        dst = f"{bundle_dir}/flow.pt"
        if not os.path.exists(dst):
            os.symlink(f"{PRETRAINED_DIR}/flow.pt", dst)


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
    codes = list_regime_codes()
    print(f"=== {len(codes)} regimes found ===", flush=True)

    for code in codes:
        print(f"\n=== [{code}] resolving checkpoints ===", flush=True)
        llm_ckpt = resolve_llm(code)
        if llm_ckpt is None:
            print(f"  [{code}] no llm checkpoint, skipping entirely", flush=True)
            manifest[code] = {"status": "skipped_no_llm"}
            continue
        flow_ckpt = resolve_flow(code)
        print(f"  llm={llm_ckpt}", flush=True)
        print(f"  flow={'(none found -- only original-flow variant)' if flow_ckpt is None else flow_ckpt}", flush=True)

        pt = pick_prompt_and_target(code)
        if pt is None:
            print(f"  [{code}] no usable kaldi dev prompt/target, skipping", flush=True)
            manifest[code] = {"status": "skipped_no_data"}
            continue
        print(f"  prompt={pt['prompt_wav']}", flush=True)
        print(f"  target_text={pt['target_text']}", flush=True)

        manifest[code] = {
            "status": "attempted",
            "llm_checkpoint": llm_ckpt,
            "flow_checkpoint": flow_ckpt,
            "prompt_wav": pt["prompt_wav"],
            "prompt_text": pt["prompt_text"],
            "target_text": pt["target_text"],
            "variants": {},
        }

        variants = [("original_flow", None)]
        if flow_ckpt is not None:
            variants.append(("ft_flow", flow_ckpt))

        for variant_name, fckpt in variants:
            bundle_dir = f"{WORK_ROOT}/{code}_{variant_name}"
            out_path = f"{OUT_DIR}/{code}_{variant_name}.wav"
            try:
                print(f"  === [{code}/{variant_name}] building bundle ===", flush=True)
                build_bundle(bundle_dir, llm_ckpt, fckpt)
                print(f"  === [{code}/{variant_name}] synthesizing ===", flush=True)
                result = synthesize(bundle_dir, pt["prompt_wav"], pt["prompt_text"], pt["target_text"], out_path)
                if result["ok"]:
                    print(f"  === [{code}/{variant_name}] SAVED: {out_path} ({result['duration']:.2f}s, peak={result['peak']:.4f}) ===", flush=True)
                else:
                    print(f"  === [{code}/{variant_name}] FAILED: {result['error']} ===", flush=True)
                manifest[code]["variants"][variant_name] = result
            except Exception as e:
                print(f"  === [{code}/{variant_name}] EXCEPTION: {e} ===", flush=True)
                traceback.print_exc()
                manifest[code]["variants"][variant_name] = {"ok": False, "error": str(e)}

            with open(MANIFEST_PATH, "w") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n=== all regimes done ===", flush=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
