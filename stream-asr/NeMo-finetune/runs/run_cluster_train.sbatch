#!/bin/bash
#SBATCH --job-name=nemotron-clusters
#SBATCH --account=aih4a_udutech
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=23:45:00
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --output=/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/logs/clusters_train-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/logs/clusters_train-%j.out

source /leonardo_work/AIH4A_udutech/nemo_venv/bin/activate
module load cuda/12.2  # provides libnvvm.so for numba-cuda (RNNT loss JIT compilation)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SECRETS_FILE="/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-asr/NeMo-finetune/.env"
if [ -f "$SECRETS_FILE" ]; then set -a; source "$SECRETS_FILE"; set +a; fi
if [ -z "${WANDB_API_KEY:-}" ]; then echo "ERROR: WANDB_API_KEY not set" >&2; exit 1; fi

export HF_HOME="/leonardo_work/AIH4A_udutech/hf_cache"
export HF_HUB_OFFLINE=1
export NO_TORCH_COMPILE=1
export WANDB_MODE=offline  # compute nodes have no internet; sync later from a login node via `wandb sync`
export WANDB_DIR="/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/wandb"
mkdir -p "$WANDB_DIR"
export NUMBA_CACHE_DIR="/leonardo_work/AIH4A_udutech/numba_cache"

cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-asr/NeMo
CONFIG_DIR=/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/configs
DONE_DIR=/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/checkpoints/clusters/_done
mkdir -p "$DONE_DIR"

# 5 clusters, exactly 5 nodes -- all run fully in parallel, one node each, no queueing needed.
# Each step uses ntasks-per-node=4 (one task per GPU) + cpus-per-task=8, matching the
# combined job's proven-working pattern. Confirmed empirically that ntasks-per-node=1 +
# devices=4 does NOT work here -- PyTorch Lightning's SLURM plugin requires devices to
# match ntasks-per-node exactly and does not self-spawn workers within a single task.
# Resumable at two levels: this outer loop skips clusters with a _done marker (job-level resume
# across resubmits), and NeMo's own exp_manager.resume_if_exists picks up mid-training state
# for whatever was still running when a job hit its wall-clock limit.
CLUSTERS="afroasiatic east_africa_bantu southern_central_bantu west_africa_niger_congo other_languages"

for name in $CLUSTERS; do
    if [ -f "$DONE_DIR/${name}.done" ]; then
        echo "=== skipping cluster (already done): $name ==="
        continue
    fi
    echo "=== starting cluster: $name ==="
    srun --exclusive -N1 -n4 --ntasks-per-node=4 --gres=gpu:4 --cpus-per-task=8 \
        bash /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-asr/NeMo-finetune/runs/train_step_wrapper.sh \
        "$CONFIG_DIR" "cluster_${name}_config.yaml" \
        > "/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/logs/cluster_${name}.log" 2>&1 &
done
wait

# Mark clusters done only if the job actually reached max_steps successfully (checked via the
# per-cluster log for a clean finish rather than assuming -- avoids marking a crashed run done).
for name in $CLUSTERS; do
    log="/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/logs/cluster_${name}.log"
    if [ -f "$log" ] && grep -q "Trainer.fit. stopped" "$log" 2>/dev/null; then
        touch "$DONE_DIR/${name}.done"
    fi
done

# normal QOS has no account-wide TRES cap (unlike boost_qos_lprod, which caps the whole
# account at 8 nodes/32 GPUs at once and would otherwise serialize this against the combined
# and individual jobs) but the partition's MaxTime=24h applies, cutting us off mid-training.
# Self-resubmit if any cluster isn't done yet -- NeMo's exp_manager.resume_if_exists and the
# _done markers above make this safe to repeat indefinitely without redoing finished work.
REMAINING=0
for name in $CLUSTERS; do
    [ -f "$DONE_DIR/${name}.done" ] || REMAINING=$((REMAINING + 1))
done
if [ "$REMAINING" -gt 0 ]; then
    echo "=== $REMAINING cluster(s) still incomplete, resubmitting for another 24h ==="
    sbatch "$0"
fi
