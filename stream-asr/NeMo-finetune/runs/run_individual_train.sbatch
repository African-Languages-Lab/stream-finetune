#!/bin/bash
#SBATCH --job-name=nemotron-individual
#SBATCH --account=aih4a_udutech
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=23:45:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --output=/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/logs/individual_train-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/logs/individual_train-%j.out

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
DONE_DIR=/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/checkpoints/individual/_done
mkdir -p "$DONE_DIR"

# 41 languages, 8 nodes -- language-level parallelism with continuous slot reuse: up to 8 run
# concurrently, and as soon as one finishes, the next queued language starts on that freed node.
# Each step uses ntasks-per-node=4 (one task per GPU) + cpus-per-task=8, matching the combined
# job's proven-working pattern. Confirmed empirically that ntasks-per-node=1 + devices=4 does
# NOT work here -- PyTorch Lightning's SLURM plugin requires devices to match ntasks-per-node
# exactly and does not self-spawn workers within a single task.
# Resumable at two levels: this loop skips languages with a _done marker on resubmit, and
# NeMo's exp_manager.resume_if_exists picks up mid-training state for whatever was still
# running when the job hit its wall-clock limit.
LANGS="rw-RW ar-AR sw-KE en-GH ha-NG ig-NG st-ZA yo-NG ts-ZA tn-BW zu-ZA nd-ZW bm-ML ff-SN ki-KE ny-MW xh-ZA so-SO tw-GH ve-ZA am-ET lg-UG ti-ER or-KE ss-SZ af-ZA bem-ZM nso-ZA ee-GH mg-MG ln-CD sn-ZW kr-NG fon-BJ ber-MA wo-SN kri-SL en-ZA umb-AO en-UG en-NG"
MAX_PARALLEL=8

train_one() {
    local tl=$1
    local log="/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/logs/individual_${tl}.log"
    echo "=== starting individual: $tl ==="
    srun --exclusive -N1 -n4 --ntasks-per-node=4 --gres=gpu:4 --cpus-per-task=8 \
        bash /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-asr/NeMo-finetune/runs/train_step_wrapper.sh \
        "$CONFIG_DIR" "individual_${tl}_config.yaml" \
        > "$log" 2>&1
    if grep -q "Trainer.fit. stopped" "$log" 2>/dev/null; then
        touch "$DONE_DIR/${tl}.done"
    fi
    echo "=== finished individual: $tl ==="
}

for tl in $LANGS; do
    if [ -f "$DONE_DIR/${tl}.done" ]; then
        echo "=== skipping (already done): $tl ==="
        continue
    fi
    while [ "$(jobs -r | wc -l)" -ge "$MAX_PARALLEL" ]; do
        wait -n
    done
    train_one "$tl" &
done
wait

# normal QOS has no account-wide TRES cap (unlike boost_qos_lprod, which caps the whole
# account at 8 nodes/32 GPUs at once and would otherwise serialize this against the combined
# and cluster jobs) but the partition's MaxTime=24h applies, cutting us off mid-run for
# whichever languages hadn't finished. Self-resubmit if any language isn't done yet --
# NeMo's exp_manager.resume_if_exists and the _done markers above make this safe to repeat
# indefinitely without redoing finished languages.
REMAINING=0
for tl in $LANGS; do
    [ -f "$DONE_DIR/${tl}.done" ] || REMAINING=$((REMAINING + 1))
done
if [ "$REMAINING" -gt 0 ]; then
    echo "=== $REMAINING language(s) still incomplete, resubmitting for another 24h ==="
    sbatch "$0"
fi
