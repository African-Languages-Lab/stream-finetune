#!/bin/bash
#SBATCH --job-name=try_new_models
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/try_new_%j.out
#SBATCH --error=logs/try_new_%j.err

source /leonardo_work/AIH4A_udutech/cosyvoice_venv/bin/activate
module load cuda/12.2
cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune
python3 scripts/try_new_models.py \
  --refs voice_selection/eval14_refs.json \
  --out  voice_selection/eval14_results.json \
  --wavdir voice_selection/eval14_gen \
  --langs  swati bambara kanuri fon krio afrikaans bemba kikuyu ndebele kinyarwanda shona wolof yoruba \
  --codes  ss-SZ bm-ML  kr-NG  fon-BJ kri-SL af-ZA     bem-ZM ki-KE  nd-ZW   rw-RW       sn-ZW wo-SN  yo-NG
