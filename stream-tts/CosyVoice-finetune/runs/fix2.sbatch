#!/bin/bash
#SBATCH --job-name=fix2
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/fix2_%j.out
#SBATCH --error=logs/fix2_%j.err
source /leonardo_work/AIH4A_udutech/cosyvoice_venv/bin/activate
module load cuda/12.2
cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune
python3 scripts/try_new_models.py \
  --refs voice_selection/fix2_refs.json --out voice_selection/fix2_results.json \
  --wavdir voice_selection/fix2_gen \
  --workdir /leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/fix2 \
  --langs tswana igbo --codes tn-BW ig-NG
