#!/bin/bash
#SBATCH --job-name=cosyft-extract
#SBATCH --account=aih4a_udutech
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-299%24
#SBATCH --output=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/logs/extract_%A_%a.out
#SBATCH --error=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/logs/extract_%A_%a.out

# Sized from the validated smoke test (shard of 2571 utterances: ~2min embedding [CPU] +
# ~10min speech token [GPU] = ~0.233s/utt token-bound). 300 shards over 12.87M utterances is
# ~42.9k utterances/shard, ~3.3h/shard at that rate -- well inside this job's 8h budget even
# with a safety margin, and %24 throttles to 24 concurrent shards (~6 nodes at 4 GPUs each)
# so this doesn't try to dominate the cluster alongside the already-running Nemotron ASR jobs.
# Each array task is independently idempotent (shard_NNNNN.done marker), so a requeued/failed
# task is safe to simply resubmit on its own -- no self-resubmission needed here since each
# shard comfortably fits in one wall-time slot.
source /leonardo_work/AIH4A_udutech/cosyvoice_venv/bin/activate
module load cuda/12.2

cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune/scripts
python3 02_extract_features_shard.py --shard-idx "$SLURM_ARRAY_TASK_ID" --num-shards 300 --num-thread 8
