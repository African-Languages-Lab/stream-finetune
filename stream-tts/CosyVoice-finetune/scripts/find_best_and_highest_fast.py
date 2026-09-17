"""
Fast best/highest checkpoint finder for llm and flow, across many regimes at once.
"Highest" = latest checkpoint by mtime (cheap: directory listing + stat, no loading).
"Best" = the checkpoint file whose mtime is closest to the wall-clock time of the lowest
CV/loss point in that stage's tensorboard log -- avoids loading every checkpoint's tensor
content just to read its embedded step (which is what made the earlier per-regime searches
take minutes each; at 31 regimes that doesn't scale). This trades a small amount of matching
precision for a roughly 100x speedup, acceptable since "best" only needs to land near the true
optimum, not hit it exactly.
"""
import glob
import json
import os
import sys

from tensorboard.backend.event_processing import event_accumulator

CKPT_ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual"


def best_cv_time(tb_dir):
    files = sorted(glob.glob(f"{tb_dir}/events.out.tfevents.*"))
    points = {}
    for fpath in files:
        try:
            ea = event_accumulator.EventAccumulator(fpath, size_guidance={"scalars": 0})
            ea.Reload()
        except Exception:
            continue
        for tag in ea.Tags().get("scalars", []):
            if tag == "CV/loss":
                for ev in ea.Scalars(tag):
                    points[ev.step] = (ev.value, ev.wall_time)
    if not points:
        return None
    best_step = min(points, key=lambda s: points[s][0])
    return points[best_step][1]  # wall_time


def highest(model_dir):
    pts = [p for p in glob.glob(f"{model_dir}/*.pt") if os.path.basename(p) != "init.pt"]
    if not pts:
        return None
    return max(pts, key=os.path.getmtime)


def closest_by_mtime(model_dir, target_time):
    pts = [p for p in glob.glob(f"{model_dir}/*.pt") if os.path.basename(p) != "init.pt"]
    if not pts:
        return None
    return min(pts, key=lambda p: abs(os.path.getmtime(p) - target_time))


def main():
    regimes = sys.argv[1:]
    result = {}
    for lang in regimes:
        base = f"{CKPT_ROOT}/individual_{lang}"
        entry = {}
        for stage in ["llm", "flow"]:
            model_dir = f"{base}/{stage}"
            tb_dir = f"{base}/tensorboard/{stage}"
            h = highest(model_dir)
            t = best_cv_time(tb_dir)
            b = closest_by_mtime(model_dir, t) if t else h
            entry[stage] = {"highest": h, "best": b}
            print(f"{lang} {stage}: highest={os.path.basename(h) if h else None} best={os.path.basename(b) if b else None}", flush=True)
        result[lang] = entry
    with open("/tmp/best_highest_result.json", "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
