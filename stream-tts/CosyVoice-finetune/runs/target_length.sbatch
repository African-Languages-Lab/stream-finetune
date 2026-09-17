#!/bin/bash
#SBATCH --job-name=tlen
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --gres=gpu:1 --time=01:30:00
#SBATCH --output=logs/tlen_%j.out
#SBATCH --error=logs/tlen_%j.err
source /leonardo_work/AIH4A_udutech/cosyvoice_venv/bin/activate
module load cuda/12.2
cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune
REF="$PWD/voice_selection/final/hausa__male.wav"
TXT=$(python3 -c "import json;print(json.load(open('voice_selection/final/prompts.json'))['hausa']['male']['prompt_text'])")
python3 scripts/target_length.py "$REF" "$TXT" voice_selection/tlen_out
