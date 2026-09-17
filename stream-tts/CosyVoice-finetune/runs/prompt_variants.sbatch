#!/bin/bash
#SBATCH --job-name=pvar
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/pvar_%j.out
#SBATCH --error=logs/pvar_%j.err
source /leonardo_work/AIH4A_udutech/cosyvoice_venv/bin/activate
module load cuda/12.2
cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune
REF="$PWD/voice_selection/final/hausa__male.wav"
TXT=$(python3 -c "import json;print(json.load(open('voice_selection/final/prompts.json'))['hausa']['male']['prompt_text'])")
python3 scripts/prompt_variants.py hausa ha-NG "$REF" "$TXT" \
  "Sannu da safe, yaya kake?" voice_selection/pvar_out
