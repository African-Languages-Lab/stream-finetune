import glob
import os

import torch


def latest_step(model_dir):
    pts = [p for p in glob.glob(f"{model_dir}/*.pt") if os.path.basename(p) != "init.pt"]
    if not pts:
        return None, None
    latest = max(pts, key=os.path.getmtime)
    try:
        sd = torch.load(latest, map_location="cpu", weights_only=True)
        step = sd.get("step", 0) if isinstance(sd, dict) else 0
    except Exception:
        return None, None
    return step, latest


def scan(base, label):
    results = []
    for path in sorted(glob.glob(f"{base}/individual_*/hifigan")):
        lang = os.path.basename(os.path.dirname(path))[len("individual_"):]
        step, p = latest_step(path)
        if step is not None:
            results.append((lang, step, p))
    results.sort(key=lambda x: -x[1])
    print(f"=== {label} hifigan, by latest-checkpoint step ===", flush=True)
    for lang, step, p in results[:15]:
        print(f"  {lang:10s} step={step:>8,d}  {p}", flush=True)
    print(flush=True)
    return results


scan("/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints_archived_20260818/individual", "ARCHIVED (old run)")
scan("/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual", "CURRENT (new restart)")
