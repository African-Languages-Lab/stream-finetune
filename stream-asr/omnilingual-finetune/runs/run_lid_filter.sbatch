#!/bin/bash
#SBATCH --job-name=omni_lid
#SBATCH --partition=lrd_all_serial
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/leonardo_scratch/large/userexternal/atsado00/omni_ft/eng_scan/lid_%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/atsado00/omni_ft/eng_scan/lid_%j.err

# Classify every transcript in every non-English language and record the English rows.
# CPU-only text work, so this belongs on the serial partition, not on a GPU node.
set -euo pipefail
SCRIPTS=/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-asr/omnilingual-finetune/scripts
OUT=/leonardo_scratch/large/userexternal/atsado00/omni_ft/english_drops
mkdir -p "$OUT"

# One process per language, four at a time: each loads its own ~1.1 GB copy of the LID model.
python3 - "$SCRIPTS" <<'PY' > "$OUT/langs.txt"
import sys, os
sys.path.insert(0, sys.argv[1])
from lang_map import LANG_MAP, token_for
print("\n".join(c for c in sorted(LANG_MAP) if token_for(c) != "eng_Latn"))
PY

xargs -P 4 -I{} python3 "$SCRIPTS/lid_filter.py" --out "$OUT" --langs {} < "$OUT/langs.txt" \
  | grep -v '^lang \|^$\|^decisions\|^total English' > "$OUT/per_lang_report.txt" || true

sort -t$'\t' -k1,1 "$OUT/per_lang_report.txt" 2>/dev/null | sort -k6 -gr | tee "$OUT/report_sorted.txt"
echo
wc -l "$OUT"/*.english.tsv | tail -1
