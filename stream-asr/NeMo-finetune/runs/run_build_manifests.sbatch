#!/bin/bash
#SBATCH --job-name=nemotron-manifests-v2
#SBATCH --account=aih4a_udutech
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/logs/manifests_v2-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/logs/manifests_v2-%j.out

source /leonardo_work/AIH4A_udutech/nemo_venv/bin/activate
python3 /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-asr/NeMo-finetune/scripts/build_manifests.py
