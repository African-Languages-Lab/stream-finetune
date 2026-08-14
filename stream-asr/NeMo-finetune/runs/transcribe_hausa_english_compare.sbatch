#!/bin/bash
#SBATCH --job-name=transcribe-ha-en
#SBATCH --account=aih4a_udutech
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/logs/transcribe_ha_en-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/logs/transcribe_ha_en-%j.out

source /leonardo_work/AIH4A_udutech/nemo_venv/bin/activate
module load cuda/12.2
export HF_HUB_OFFLINE=1

cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-asr/NeMo-finetune/scripts
python3 transcribe_hausa_english_compare.py
