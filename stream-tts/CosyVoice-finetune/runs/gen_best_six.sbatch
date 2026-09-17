#!/bin/bash
#SBATCH --job-name=genbest6
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --account=AIH4A_udutech
#SBATCH --output=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/inference_best_all_languages/logs/%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/inference_best_all_languages/logs/%j.err
module load cuda/12.2
source /leonardo_work/AIH4A_udutech/cosyvoice_venv/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export REGIMES=english,ha-NG,ig-NG,tw-GH,ee-GH,yo-NG
cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune
python scripts/generate_best_all_languages.py
