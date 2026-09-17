#!/bin/bash
#SBATCH --job-name=omni-ft
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --account=AIH4A_udutech
#SBATCH --output=/leonardo_scratch/large/userexternal/atsado00/omni_ft/logs/ft_%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/atsado00/omni_ft/logs/ft_%j.err
set -e
module load cuda/12.2
source /leonardo_work/AIH4A_udutech/omni_venv/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
REPO=/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-asr
OUT=/leonardo_scratch/large/userexternal/atsado00/omni_ft/runs/combined_ctc_300m
mkdir -p "$OUT"
cd $REPO/omnilingual-asr
srun python -m workflows.recipes.wav2vec2.asr "$OUT" \
  --config-file $REPO/omnilingual-finetune/configs/combined-ctc-finetune.yaml
