#!/bin/bash
#SBATCH --job-name=cosyft-smoke-extract
#SBATCH --account=aih4a_udutech
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --output=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/logs/smoke_extract-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/logs/smoke_extract-%j.out

source /leonardo_work/AIH4A_udutech/cosyvoice_venv/bin/activate
module load cuda/12.2

cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune/scripts
time python3 02_extract_features_shard.py --shard-idx 0 --num-shards 5000 --num-thread 16
