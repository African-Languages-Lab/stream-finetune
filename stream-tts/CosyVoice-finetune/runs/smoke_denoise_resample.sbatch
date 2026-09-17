#!/bin/bash
#SBATCH --job-name=cosyft-denoise-smoke
#SBATCH --account=aih4a_udutech
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --output=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/logs/denoise_smoke_%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/logs/denoise_smoke_%j.out

source /leonardo_work/AIH4A_udutech/cosyvoice_denoise_venv/bin/activate
cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune/scripts
# smoke test: 1 shard out of 3000 (~4284 utterances) to measure real throughput before sizing
# the full array job over all 12.85M utterances
python3 00_denoise_resample_shard.py --shard-idx 0 --num-shards 3000
