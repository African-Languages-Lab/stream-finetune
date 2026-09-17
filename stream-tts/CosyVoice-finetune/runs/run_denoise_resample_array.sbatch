#!/bin/bash
#SBATCH --job-name=cosyft-denoise
#SBATCH --account=aih4a_udutech
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --array=0-149%20
#SBATCH --output=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/logs/denoise_%A_%a.out
#SBATCH --error=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/logs/denoise_%A_%a.out

# Sized from the validated smoke test (shard of 4281 utterances: ~9.5 utt/s CPU-only, 12.5% of
# files needed real denoising, 0 failures). 150 shards over 12.85M utterances is ~85.7k
# utts/shard, ~2.5h/shard at that rate -- comfortably inside this job's 4h budget. %20 throttles
# concurrency since this runs on boost_usr_prod (CPU-only via --gres=gpu:0 -- our account has no
# dcgp_usr_prod authorization) so it doesn't compete too hard with GPU-bound training/inference
# jobs for node allocation. Each array task is independently idempotent (shard_NNNNN.done
# marker, and per-utterance the output file's existence is also checked), so a
# failed/requeued task is safe to just resubmit on its own.
source /leonardo_work/AIH4A_udutech/cosyvoice_denoise_venv/bin/activate

cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune/scripts
python3 00_denoise_resample_shard.py --shard-idx "$SLURM_ARRAY_TASK_ID" --num-shards 150
