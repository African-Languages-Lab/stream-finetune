"""
One best-quality generation per ready individual-language regime: true best llm + true best
flow, each independently chosen as whichever of (current run, archived run) has the lower
CV/loss (tensorboard-matched to the closest-mtime checkpoint), not "recent-preferred" like the
archived-vs-recent comparison scripts. Original pretrained vocoder throughout (hifigan
finetuning is retired) and the original/standard CosyVoice tokenizer -- no custom text
handling. A language with no usable llm checkpoint yet (still training, or never produced one)
is skipped automatically.

Reference prompt+target are pulled from that language's own kaldi dev data, excluding
Waxal_NLP_* sources -- confirmed (see conversation) to be short code-mixed template/exercise
sentences ("Àpẹẹrẹ Triangle Wà Lórí Circle") rather than natural speech, responsible for a
near-silent, prematurely-truncated Yoruba sample in an earlier run despite passing the
duration/length filters. Preferring natural sources whenever any exist applies to every
language here, not just Yoruba.
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
WORK_ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/bundle_best_all_languages"
OUT_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/inference_best_all_languages"
MANIFEST_PATH = f"{OUT_DIR}/manifest.json"
os.makedirs(OUT_DIR, exist_ok=True)


def list_regime_codes():
    codes = []
    for d in sorted(glob.glob(f"{CURRENT_ROOT}/individual_*")):
        codes.append(os.path.basename(d)[len("individual_"):])
    # REGIMES lets a run be scoped to a subset without editing this file; unset = all, as before
    only = [c.strip() for c in os.environ.get("REGIMES", "").split(",") if c.strip()]
    if only:
        codes = [c for c in codes if c in only]
    return codes


def highest_checkpoint(stage_dir):
    files = [p for p in glob.glob(f"{stage_dir}/*.pt") if os.path.basename(p) != "init.pt"]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def best_checkpoint_with_loss(stage_dir):
    """(checkpoint_path, cv_loss) for the CV/loss minimum, tensorboard-matched to the
    closest-mtime checkpoint; (None, None) if no tensorboard data or no checkpoints."""
    tb_dir = stage_dir.replace("/llm", "/tensorboard/llm").replace("/flow", "/tensorboard/flow")
    try:
        from tensorboard.backend.event_processing import event_accumulator
        ea = event_accumulator.EventAccumulator(tb_dir, size_guidance={"scalars": 0})
        ea.Reload()
        events = ea.Scalars("CV/loss")
        best_event = min(events, key=lambda e: e.value)
        files = [p for p in glob.glob(f"{stage_dir}/*.pt") if os.path.basename(p) != "init.pt"]
        if not files:
            return None, None
        ckpt = min(files, key=lambda p: abs(os.path.getmtime(p) - best_event.wall_time))
        return ckpt, best_event.value
    except Exception:
        return None, None


def resolve_true_best(current_dir, archived_dir):
    """Whichever of (current, archived) has the lower CV/loss wins, independent of recency.
    Falls back to highest-by-mtime (current preferred) if neither has tensorboard data."""
    candidates = []
    cur_ckpt, cur_loss = best_checkpoint_with_loss(current_dir)
    if cur_ckpt is not None:
        candidates.append((cur_loss, cur_ckpt, "current"))
    arc_ckpt, arc_loss = best_checkpoint_with_loss(archived_dir)
    if arc_ckpt is not None:
        candidates.append((arc_loss, arc_ckpt, "archived"))
    if candidates:
        candidates.sort(key=lambda c: c[0])
        loss, ckpt, source = candidates[0]
        return ckpt, source, loss
    ckpt = highest_checkpoint(current_dir)
    if ckpt is not None:
        return ckpt, "current_highest_fallback", None
    ckpt = highest_checkpoint(archived_dir)
    if ckpt is not None:
        return ckpt, "archived_highest_fallback", None
    return None, None, None


def resolve_llm(code):
    return resolve_true_best(f"{CURRENT_ROOT}/individual_{code}/llm", f"{ARCHIVE_ROOT}/individual_{code}/llm")


def resolve_flow(code):
    return resolve_true_best(f"{CURRENT_ROOT}/individual_{code}/flow", f"{ARCHIVE_ROOT}/individual_{code}/flow")


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
    for asset in ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "CosyVoice-BlankEN", "hift.pt"]:
        dst = f"{bundle_dir}/{asset}"
        if not os.path.exists(dst):
            os.symlink(f"{PRETRAINED_DIR}/{asset}", dst)
    clean(llm_ckpt, f"{bundle_dir}/llm.pt")
    clean(flow_ckpt, f"{bundle_dir}/flow.pt")


def synthesize(bundle_dir, prompt_wav, prompt_text, target_text, out_path):
    """Render the same prompt+target both ways, from one loaded model.

    zero_shot     sends the reference transcript. This is the original behaviour and what
                  the live endpoint does for 47 of its 48 preset voices.
    cross_lingual sends no transcript; the voice carries through the flow prompt and the
                  speaker embedding only.

    Hyperparameters are untouched -- both are stock CosyVoice3 calls with default arguments,
    so n_timesteps stays at the upstream 10 and nothing is monkeypatched.
    """
    model = CosyVoice3(bundle_dir, fp16=False)
    out = {}
    for mode, path in (("zero_shot", out_path),
                       ("cross_lingual", out_path.replace("_best.wav", "_best_crosslingual.wav"))):
        last_err = None
        for attempt in range(4):
            try:
                if mode == "zero_shot":
                    results = list(model.inference_zero_shot(
                        target_text, prompt_text, prompt_wav, stream=False))
                else:
                    # cross-lingual needs the instruction preamble on the TARGET text; the
                    # kaldi prompt_text carries its own marker but that is not sent here.
                    # Without it the LM emits a near-empty token sequence and the flow
                    # decoder dies with "Calculated padded input size per channel: (3)".
                    results = list(model.inference_cross_lingual(
                        "You are a helpful assistant.<|endofprompt|>" + target_text,
                        prompt_wav, stream=False))
                audio = results[0]["tts_speech"]
                torchaudio.save(path, audio, model.sample_rate)
                out[mode] = {"ok": True,
                             "duration": audio.shape[1] / model.sample_rate,
                             "peak": audio.abs().max().item(),
                             "file": os.path.basename(path)}
                break
            except RuntimeError as e:
                last_err = str(e)
        else:
            out[mode] = {"ok": False, "error": last_err}
    del model
    zs = out["zero_shot"]
    return {"ok": zs.get("ok", False), "duration": zs.get("duration"),
            "peak": zs.get("peak"), "error": zs.get("error"), "modes": out}


def main():
    manifest = {}
    codes = list_regime_codes()
    print(f"=== {len(codes)} candidate regimes ===", flush=True)

    for code in codes:
        print(f"\n=== [{code}] resolving best checkpoints ===", flush=True)
        llm_ckpt, llm_source, llm_loss = resolve_llm(code)
        if llm_ckpt is None:
            print(f"  [{code}] no llm checkpoint at all yet, skipping (not ready)", flush=True)
            manifest[code] = {"status": "skipped_not_ready_no_llm"}
            continue
        flow_ckpt, flow_source, flow_loss = resolve_flow(code)
        if flow_ckpt is None:
            print(f"  [{code}] no flow checkpoint at all yet, skipping (not ready)", flush=True)
            manifest[code] = {"status": "skipped_not_ready_no_flow"}
            continue

        print(f"  best_llm={llm_ckpt} (source={llm_source}, cv_loss={llm_loss})", flush=True)
        print(f"  best_flow={flow_ckpt} (source={flow_source}, cv_loss={flow_loss})", flush=True)

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
            "llm_source": llm_source,
            "llm_cv_loss": llm_loss,
            "flow_checkpoint": flow_ckpt,
            "flow_source": flow_source,
            "flow_cv_loss": flow_loss,
            "prompt_wav": pt["prompt_wav"],
            "prompt_text": pt["prompt_text"],
            "target_text": pt["target_text"],
        }

        bundle_dir = f"{WORK_ROOT}/{code}"
        out_path = f"{OUT_DIR}/{code}_best.wav"
        try:
            print(f"  === [{code}] building bundle ===", flush=True)
            build_bundle(bundle_dir, llm_ckpt, flow_ckpt)
            print(f"  === [{code}] synthesizing ===", flush=True)
            result = synthesize(bundle_dir, pt["prompt_wav"], pt["prompt_text"], pt["target_text"], out_path)
            if result["ok"]:
                print(f"  === [{code}] SAVED: {out_path} ({result['duration']:.2f}s, peak={result['peak']:.4f}) ===", flush=True)
            else:
                print(f"  === [{code}] FAILED: {result['error']} ===", flush=True)
            manifest[code]["result"] = result
        except Exception as e:
            print(f"  === [{code}] EXCEPTION: {e} ===", flush=True)
            traceback.print_exc()
            manifest[code]["result"] = {"ok": False, "error": str(e)}

        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n=== all regimes done ===", flush=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
