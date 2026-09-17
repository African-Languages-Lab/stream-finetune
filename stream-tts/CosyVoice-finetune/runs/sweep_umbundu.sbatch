#!/bin/bash
#SBATCH --job-name=sweep_umb
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --account=AIH4A_udutech
#SBATCH --output=/leonardo_scratch/large/userexternal/atsado00/ckptsweep/logs/%x_%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/atsado00/ckptsweep/logs/%x_%j.err
module load cuda/12.2
source /leonardo_work/AIH4A_udutech/cosyvoice_venv/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune
python scripts/ckpt_sweep.py --lang umbundu \
  --llms epoch_1_whole.pt,epoch_3_whole.pt,epoch_5_whole.pt,epoch_7_whole.pt \
  --flows epoch_50_whole.pt,epoch_91_whole.pt,epoch_150_whole.pt \
  --out /leonardo_scratch/large/userexternal/atsado00/ckptsweep/umbundu --reps 2
