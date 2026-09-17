#!/bin/bash
#SBATCH --job-name=verify_prompts
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/verify_prompts_%j.out
#SBATCH --error=logs/verify_prompts_%j.err

source /leonardo_work/AIH4A_udutech/cosyvoice_venv/bin/activate
module load cuda/12.2
set -a; source /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/.env; set +a
cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune
python3 scripts/verify_reference_prompts.py \
  --candidates "$1" --out "$2" ${3:+--langs $3}
