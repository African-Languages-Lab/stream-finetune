import json
from pathlib import Path

MANIFEST_DIR = Path("/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/manifests_clean")
OUT_DIR = Path("/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/configs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

summary = json.load(open(MANIFEST_DIR / "manifest_build_summary.json"))
agg = {}
for bucket, v in summary.items():
    tl = v["target_lang"]
    agg[tl] = agg.get(tl, 0.0) + v["hours_written"]

alpha = 0.3
raw = {tl: h**alpha for tl, h in agg.items()}
total = sum(raw.values())
weights = {tl: raw[tl] / total for tl in raw}

# --- Lhotse input_cfg for training data blend ---
train_cfg = []
for tl, w in sorted(weights.items(), key=lambda x: -x[1]):
    train_cfg.append({
        "type": "nemo",
        "manifest_filepath": str(MANIFEST_DIR / f"{tl}_train.jsonl"),
        "weight": round(w, 6),
        "tags": {"target_lang": tl, "prompt_mode": "unified"},
    })
with open(OUT_DIR / "train_blend.yaml", "w") as f:
    f.write("input_cfg:\n")
    for e in train_cfg:
        f.write(f"  - type: {e['type']}\n")
        f.write(f"    manifest_filepath: {e['manifest_filepath']}\n")
        f.write(f"    weight: {e['weight']}\n")
        f.write(f"    tags: {{target_lang: {e['tags']['target_lang']}, prompt_mode: {e['tags']['prompt_mode']}}}\n")

# --- validation input_cfg: equal-weight list of all dev manifests that exist and are non-empty ---
dev_cfg = []
for tl in agg:
    p = MANIFEST_DIR / f"{tl}_dev.jsonl"
    if p.exists() and p.stat().st_size > 0:
        dev_cfg.append(tl)
with open(OUT_DIR / "dev_blend.yaml", "w") as f:
    f.write("input_cfg:\n")
    for tl in sorted(dev_cfg):
        f.write(f"  - type: nemo\n")
        f.write(f"    manifest_filepath: {MANIFEST_DIR / f'{tl}_dev.jsonl'}\n")
        f.write(f"    weight: 1.0\n")
        f.write(f"    tags: {{target_lang: {tl}}}\n")

print(f"train languages: {len(train_cfg)}, dev languages: {len(dev_cfg)}")
print("weight range:", min(weights.values()), "to", max(weights.values()))
