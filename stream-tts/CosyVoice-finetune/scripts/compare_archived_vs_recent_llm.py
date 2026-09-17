"""
Archived-ft-llm vs recent-ft-llm comparison for Igbo, Hausa, Yoruba, Twi, Ewe.

For each language: two variants, "archived_llm" and "recent_llm", both built with the SAME
flow (the recent/current ft flow -- not archived, not original) and the SAME original
pretrained vocoder throughout (hifigan finetuning is retired). Only the llm checkpoint
differs between variants. Both variants use the identical prompt+target reference pair
(pulled from that language's own kaldi dev data), so any audible difference is attributable
to the llm alone.

"archived"/"recent" = best (CV/loss-minimum, tensorboard-matched) checkpoint in
checkpoints_archived_20260818/ vs checkpoints/ respectively, falling back to highest-by-mtime
if tensorboard data is missing. Hausa has no run under checkpoints_archived_20260818/ at all
(it was only ever trained once) -- for Hausa, "archived" falls back to the earliest
(min-mtime) checkpoint in its own current run and "recent" is that same run's best/highest,
so the comparison still reflects an earlier-vs-later llm state rather than being skipped.
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
WORK_ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/bundle_archived_vs_recent"
OUT_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/inference_archived_vs_recent"
MANIFEST_PATH = f"{OUT_DIR}/manifest.json"
os.makedirs(OUT_DIR, exist_ok=True)

LANGUAGE_CODES = ["ig-NG", "ha-NG", "yo-NG", "tw-GH", "ee-GH"]


def highest_checkpoint(stage_dir, exclude=()):
    files = [p for p in glob.glob(f"{stage_dir}/*.pt") if os.path.basename(p) not in ("init.pt", *exclude)]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def earliest_checkpoint(stage_dir, exclude=()):
    files = [p for p in glob.glob(f"{stage_dir}/*.pt") if os.path.basename(p) not in ("init.pt", *exclude)]
    if not files:
        return None
    return min(files, key=os.path.getmtime)


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


def resolve_recent_llm(code):
    stage_dir = f"{CURRENT_ROOT}/individual_{code}/llm"
    return best_checkpoint(stage_dir) or highest_checkpoint(stage_dir)


def resolve_archived_llm(code, recent_ckpt):
    """Best/highest checkpoint under the archived run; if no archived run exists at all,
    fall back to the earliest checkpoint of the CURRENT run (distinct from recent_ckpt)."""
    stage_dir = f"{ARCHIVE_ROOT}/individual_{code}/llm"
    ckpt = best_checkpoint(stage_dir) or highest_checkpoint(stage_dir)
    if ckpt is not None:
        return ckpt, "archived_run"
    current_stage_dir = f"{CURRENT_ROOT}/individual_{code}/llm"
    exclude = (os.path.basename(recent_ckpt),) if recent_ckpt else ()
    ckpt = earliest_checkpoint(current_stage_dir, exclude=exclude)
    return ckpt, "earliest_current_fallback"


def resolve_recent_flow(code):
    """The recent/current ft flow, used for BOTH variants. Falls back to original pretrained
    flow if the current run has no flow checkpoint (not expected for these 5 languages)."""
    stage_dir = f"{CURRENT_ROOT}/individual_{code}/flow"
    ckpt = best_checkpoint(stage_dir) or highest_checkpoint(stage_dir)
    return ckpt


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

    # Waxal_NLP_* sources are short template/exercise sentences (e.g. shape-naming, weather
    # templates, often code-mixed with English words like "Triangle"/"Circle") rather than
    # natural speech -- confirmed responsible for a near-silent, prematurely-truncated Yoruba
    # comparison (both llm variants collapsed identically on a Waxal-sourced prompt+target
    # pair) despite passing the duration/length filters below. Prefer non-Waxal candidates
    # whenever any exist for this language; only fall back to Waxal if it's all there is.
    def is_natural(path):
        return "Waxal_NLP" not in path

    natural = [c for c in candidates if is_natural(c[1])]
    pool = natural if natural else candidates

    prompt = next(((u, p, d, t) for u, p, d, t in pool if 7.0 <= d <= 15.0 and len(t) >= 15), None)
    if prompt is None:
        candidates_sorted = sorted(pool, key=lambda c: c[2], reverse=True)
        prompt = candidates_sorted[0] if candidates_sorted else None
    if prompt is None:
        return None

    target = next(
        ((u, p, d, t) for u, p, d, t in pool if u != prompt[0] and 60 <= len(t) <= 180),
        None,
    )
    if target is None:
        others = [c for c in pool if c[0] != prompt[0]]
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
    print(f"=== {len(LANGUAGE_CODES)} languages: {LANGUAGE_CODES} ===", flush=True)

    for code in LANGUAGE_CODES:
        print(f"\n=== [{code}] resolving checkpoints ===", flush=True)
        recent_llm = resolve_recent_llm(code)
        if recent_llm is None:
            print(f"  [{code}] no recent llm checkpoint, skipping entirely", flush=True)
            manifest[code] = {"status": "skipped_no_recent_llm"}
            continue
        archived_llm, archived_source = resolve_archived_llm(code, recent_llm)
        if archived_llm is None:
            print(f"  [{code}] no archived llm checkpoint available (even with fallback), skipping", flush=True)
            manifest[code] = {"status": "skipped_no_archived_llm"}
            continue
        recent_flow = resolve_recent_flow(code)

        print(f"  recent_llm={recent_llm}", flush=True)
        print(f"  archived_llm={archived_llm} (source={archived_source})", flush=True)
        print(f"  recent_flow (used for both variants)={'(none found -- falling back to original flow)' if recent_flow is None else recent_flow}", flush=True)

        pt = pick_prompt_and_target(code)
        if pt is None:
            print(f"  [{code}] no usable kaldi dev prompt/target, skipping", flush=True)
            manifest[code] = {"status": "skipped_no_data"}
            continue
        print(f"  prompt={pt['prompt_wav']}", flush=True)
        print(f"  target_text={pt['target_text']}", flush=True)

        manifest[code] = {
            "status": "attempted",
            "recent_llm_checkpoint": recent_llm,
            "archived_llm_checkpoint": archived_llm,
            "archived_llm_source": archived_source,
            "recent_flow_checkpoint": recent_flow,
            "prompt_wav": pt["prompt_wav"],
            "prompt_text": pt["prompt_text"],
            "target_text": pt["target_text"],
            "variants": {},
        }

        for variant_name, llm_ckpt in [("archived_llm", archived_llm), ("recent_llm", recent_llm)]:
            bundle_dir = f"{WORK_ROOT}/{code}_{variant_name}"
            out_path = f"{OUT_DIR}/{code}_{variant_name}.wav"
            try:
                print(f"  === [{code}/{variant_name}] building bundle ===", flush=True)
                build_bundle(bundle_dir, llm_ckpt, recent_flow)
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

    print("\n=== all languages done ===", flush=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
