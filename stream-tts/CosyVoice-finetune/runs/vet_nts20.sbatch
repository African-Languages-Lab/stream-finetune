#!/bin/bash
#SBATCH --job-name=vetnts20
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --account=AIH4A_udutech
#SBATCH --output=/leonardo_scratch/large/userexternal/atsado00/vet_en_ha_ig/logs/%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/atsado00/vet_en_ha_ig/logs/%j.err
module load cuda/12.2
source /leonardo_work/AIH4A_udutech/cosyvoice_venv/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VET_STEPS=20
cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune
echo "########## CROSS-LINGUAL nts20 ##########"
python scripts/vet_best_en_ha_ig.py
echo "########## ZERO-SHOT nts20 ##########"
python scripts/vet_zeroshot_ha_ig.py
