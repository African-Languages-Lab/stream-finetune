#!/bin/bash
#SBATCH --job-name=cosyft-merge
#SBATCH --account=aih4a_udutech
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=450G
#SBATCH --output=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/logs/merge_%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/logs/merge_%j.out

source /leonardo_work/AIH4A_udutech/cosyvoice_venv/bin/activate
cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune/scripts
python3 03_merge_features.py
