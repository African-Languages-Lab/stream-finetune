"""Build a balanced text corpus for tokenizer training from all *_train.jsonl manifests.
Caps each language/variety's contribution to avoid huge datasets (Kinyarwanda, Arabic,
Hausa millions of rows) drowning out small ones (Umbundu 2k rows) in subword statistics.
"""
import json
import random
from pathlib import Path

MANIFEST_DIR = Path("/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/manifests")
OUT_FILE = Path("/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/tokenizer/corpus.txt")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

CAP_PER_SOURCE = 150_000
random.seed(0)

train_files = sorted(MANIFEST_DIR.glob("*_train.jsonl"))
print(f"{len(train_files)} train manifests found")

total_lines = 0
with open(OUT_FILE, "w", encoding="utf-8") as out:
    for mf in train_files:
        lines = []
        with open(mf, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                text = rec.get("text", "").strip()
                if text:
                    lines.append(text)
        if len(lines) > CAP_PER_SOURCE:
            lines = random.sample(lines, CAP_PER_SOURCE)
        for t in lines:
            out.write(t + "\n")
        total_lines += len(lines)
        print(f"{mf.name}: {len(lines)} lines (capped from original)")

print(f"\nTotal corpus lines: {total_lines}")
print(f"Written to {OUT_FILE}")
