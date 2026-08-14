#!/bin/bash
#SBATCH --job-name=nemotron-combined
#SBATCH --account=aih4a_udutech
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=4-00:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --output=/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/logs/combined_train-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/logs/combined_train-%j.out

source /leonardo_work/AIH4A_udutech/nemo_venv/bin/activate
module load cuda/12.2  # provides libnvvm.so for numba-cuda (RNNT loss JIT compilation)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Private secrets (WANDB_API_KEY), chmod 600, not git-tracked.
SECRETS_FILE="/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-asr/NeMo-finetune/.env"
if [ -f "$SECRETS_FILE" ]; then
    set -a
    source "$SECRETS_FILE"
    set +a
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: WANDB_API_KEY is not set in $SECRETS_FILE -- fill it in before submitting." >&2
    exit 1
fi

export HF_HOME="/leonardo_work/AIH4A_udutech/hf_cache"
export HF_HUB_OFFLINE=1
export NO_TORCH_COMPILE=1

# All training results/artifacts -> $SCRATCH/nemotron_ft. Without this, wandb writes its
# local run cache to the current working directory (the NeMo repo, in the quota-constrained
# $HOME) by default -- redirect it explicitly.
export WANDB_MODE=offline  # compute nodes have no internet; sync later from a login node via `wandb sync`
export WANDB_DIR="/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/wandb"
mkdir -p "$WANDB_DIR"
export NUMBA_CACHE_DIR="/leonardo_work/AIH4A_udutech/numba_cache"

# PyTorch Lightning picks up SLURM_NTASKS/SLURM_NODEID/SLURM_PROCID automatically under srun,
# giving us DDP across nodes x devices without manual rank bookkeeping.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-asr/NeMo

# Unlike cluster/individual, combined never had self-resubmission: it's crashed from CUDA
# OOM on an outlier batch multiple times now (confirmed: 4+ rounds of progressively tighter
# batch_duration/max_duration/quadratic_duration/batch_size, still recurring occasionally)
# and, with no auto-restart, was just sitting idle after each crash until someone noticed and
# resubmitted by hand. resume_if_exists=True in the config already picks up from the last
# checkpoint on restart -- this just makes that automatic, same pattern as cluster/individual.
LOG_FILE="/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/logs/combined_train-${SLURM_JOB_ID}.out"
srun python examples/asr/speech_to_text_finetune.py \
    --config-path=/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/configs \
    --config-name=combined_train_config.yaml

if ! grep -q "Trainer.fit. stopped: .max_steps=100000. reached" "$LOG_FILE" 2>/dev/null; then
    echo "=== combined did not reach max_steps, resubmitting ==="
    sbatch "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-asr/NeMo-finetune/runs/run_combined_train.sbatch"
fi
