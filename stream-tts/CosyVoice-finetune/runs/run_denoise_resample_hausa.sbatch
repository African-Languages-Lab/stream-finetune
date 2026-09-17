#!/bin/bash
#SBATCH --job-name=cosyft-denoise-hausa
#SBATCH --account=aih4a_udutech
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --array=0-59%40
#SBATCH --output=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/logs/denoise_hausa_%A_%a.out
#SBATCH --error=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/logs/denoise_hausa_%A_%a.out

# Dedicated, higher-concurrency push for Hausa specifically (~1.87M utterances, ~749k already
# done by the general 150-shard array by the time this was submitted -- idempotent output-file
# check makes it safe to run this alongside that array without double-processing). 60 shards
# %40 concurrent (vs the general array's %20) so Hausa finishes well ahead of the rest of the
# corpus. Separate manifest dir (denoise_manifests/lang_hausa/) so shard numbering doesn't
# collide with the general array's own shard_NNNNN files.
source /leonardo_work/AIH4A_udutech/cosyvoice_denoise_venv/bin/activate

cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune/scripts
python3 00_denoise_resample_shard.py --shard-idx "$SLURM_ARRAY_TASK_ID" --num-shards 60 --language hausa
