#!/bin/bash
#SBATCH --job-name=cosyft-orchestrate
#SBATCH --account=aih4a_udutech
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/logs/orchestrate_%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/logs/orchestrate_%j.out

# Runs the CPU/IO-bound regime build + parquet packaging for all 47 regimes and submits each
# one's training job as its data becomes ready, sequentially. This is data-prep orchestration
# only (no GPU needed here) -- kept off the login node since packaging large languages
# (sw-KE: 1.29M rows, ar-AR: 2.2M rows) copies raw audio bytes per utterance and can run into
# the hours per language, well past what's reasonable to run interactively.
source /leonardo_work/AIH4A_udutech/cosyvoice_venv/bin/activate
bash /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune/runs/submit_all_regimes.sh
