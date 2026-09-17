import glob
import os
import sys

import torch


def closest_to(model_dir, target_step):
    pts = [p for p in glob.glob(f"{model_dir}/*.pt") if os.path.basename(p) != "init.pt"]
    best = None
    best_diff = None
    for p in pts:
        try:
            sd = torch.load(p, map_location="cpu", weights_only=True)
            step = sd.get("step", 0) if isinstance(sd, dict) else 0
        except Exception:
            continue
        diff = abs(step - target_step)
        print(f"  checked {os.path.basename(p)} step={step} diff={diff}", flush=True)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best = (p, step)
    return best


model_dir, target_step = sys.argv[1], int(sys.argv[2])
result = closest_to(model_dir, target_step)
print(f"CLOSEST: {result}", flush=True)
