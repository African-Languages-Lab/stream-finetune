import glob
import json
import os

import torch

CKPT_ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints"
STAGES = ["llm", "flow", "hifigan"]
TARGETS = {
    "individual": {"llm": 60000, "flow": 60000, "hifigan": 250000},
    "cluster": {"llm": 100000, "flow": 100000, "hifigan": 350000},
    "combined": {"llm": 150000, "flow": 150000, "hifigan": 500000},
}


def latest_step(model_dir):
    pts = [p for p in glob.glob(f"{model_dir}/*.pt") if os.path.basename(p) != "init.pt"]
    if not pts:
        return 0
    latest = max(pts, key=os.path.getmtime)
    try:
        sd = torch.load(latest, map_location="cpu", weights_only=True)
        return sd.get("step", 0) if isinstance(sd, dict) else 0
    except Exception:
        return -1


def report(kind, name, path):
    state_file = f"{path}/stage.state.json"
    stage_idx = 0
    if os.path.exists(state_file):
        stage_idx = json.load(open(state_file)).get("stage_idx", 0)
    stage = STAGES[stage_idx] if stage_idx < 3 else "DONE"
    if stage == "DONE":
        print(f"{kind:10s} {name:24s} DONE (all stages converged)", flush=True)
        return
    step = latest_step(f"{path}/{stage}")
    target = TARGETS[kind][stage]
    pct = 100 * step / target if target else 0
    print(f"{kind:10s} {name:24s} stage={stage:8s} step={step:>7,d}/{target:<7,d} ({pct:4.1f}%)", flush=True)


print(f"{'kind':10s} {'regime':24s} progress", flush=True)
print("=" * 70, flush=True)
for path in sorted(glob.glob(f"{CKPT_ROOT}/individual/individual_*")):
    name = os.path.basename(path)[len("individual_"):]
    report("individual", name, path)
print(flush=True)
for path in sorted(glob.glob(f"{CKPT_ROOT}/cluster/cluster_*")):
    name = os.path.basename(path)
    report("cluster", name, path)
print(flush=True)
combined_path = f"{CKPT_ROOT}/combined/combined"
if os.path.isdir(combined_path):
    report("combined", "combined", combined_path)
