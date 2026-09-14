#!/bin/bash
#SBATCH --job-name=omni_mf
#SBATCH --partition=lrd_all_serial
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/leonardo_scratch/large/userexternal/atsado00/omni_ft/eng_scan/mf_%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/atsado00/omni_ft/eng_scan/mf_%j.err
set -euo pipefail
cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-asr/omnilingual-finetune
python3 scripts/build_manifests.py \
  --out /leonardo_scratch/large/userexternal/atsado00/omni_ft/manifests \
  --english-drops /leonardo_scratch/large/userexternal/atsado00/omni_ft/english_drops
