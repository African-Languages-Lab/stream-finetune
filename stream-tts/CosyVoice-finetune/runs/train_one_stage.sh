#!/bin/bash
# Trains one CosyVoice3 component (llm, flow, or hifigan) for one regime's data, resuming
# from that stage's latest self-produced checkpoint if one exists, otherwise initializing
# from the pretrained release checkpoint. Meant to be invoked once per SLURM job (a wall-time
# signal ends the torchrun process cleanly; the caller decides whether to resubmit).
#
# Usage: train_one_stage.sh <regime_name> <model: llm|flow|hifigan> <num_gpus> <checkpoint_dir> <parquet_dir>
set -eu

REGIME=$1
MODEL=$2
NUM_GPUS=$3
CHECKPOINT_DIR=$4   # e.g. /leonardo_scratch/.../checkpoints/individual/rw-RW
PARQUET_DIR=$5      # e.g. /leonardo_scratch/.../parquet/individual_rw-RW

PRETRAINED_DIR=/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B
CONFIG=/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune/configs/cosyvoice3_sft.yaml
COSYVOICE_ROOT=/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice

# The pretrained bundle ships the vocoder generator as hift.pt, but train.py's --model
# argument for GAN (generator+discriminator) training is "hifigan" (a separate wrapper class
# in cosyvoice3.yaml that composes the hift generator with a discriminator that has no
# pretrained weights and starts fresh) -- the reference examples/*/run.sh's "$model.pt"
# naming pattern silently breaks on this ($pretrained_dir/hifigan.pt does not exist).
case "$MODEL" in
    llm) PRETRAINED_CKPT="$PRETRAINED_DIR/llm.pt" ;;
    flow) PRETRAINED_CKPT="$PRETRAINED_DIR/flow.pt" ;;
    hifigan) PRETRAINED_CKPT="$PRETRAINED_DIR/hift.pt" ;;
    *) echo "unknown model $MODEL" >&2; exit 1 ;;
esac

MODEL_DIR="$CHECKPOINT_DIR/$MODEL"
mkdir -p "$MODEL_DIR"

# Resume from this stage's own latest checkpoint if one exists (by mtime), else the
# pretrained release checkpoint. train.py has no separate "auto-resume from model_dir"
# logic -- whatever file --checkpoint points to is what gets loaded, and its embedded
# 'step'/'epoch' fields (written by save_model) set the training loop's starting point.
LATEST=$(ls -t "$MODEL_DIR"/*.pt 2>/dev/null | head -n1 || true)
if [ -n "$LATEST" ]; then
    CKPT="$LATEST"
    echo "=== resuming $REGIME/$MODEL from $CKPT ==="
else
    CKPT="$PRETRAINED_CKPT"
    echo "=== starting $REGIME/$MODEL fresh from pretrained $CKPT ==="
fi

cd "$COSYVOICE_ROOT"
# `python cosyvoice/bin/train.py` puts cosyvoice/bin/ on sys.path[0], not the repo root, so
# `import cosyvoice` fails unless the root (and third_party/Matcha-TTS, needed by the flow
# model) are on PYTHONPATH explicitly -- matches examples/*/path.sh, absolute paths since we
# already cd here.
export PYTHONPATH="$COSYVOICE_ROOT:$COSYVOICE_ROOT/third_party/Matcha-TTS:${PYTHONPATH:-}"
torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" \
    --rdzv_id="${SLURM_JOB_ID:-1}" --rdzv_backend=c10d --rdzv_endpoint=localhost:1234 \
    cosyvoice/bin/train.py \
    --train_engine torch_ddp \
    --config "$CONFIG" \
    --train_data "$PARQUET_DIR/train.data.list" \
    --cv_data "$PARQUET_DIR/dev.data.list" \
    --qwen_pretrain_path "$PRETRAINED_DIR/CosyVoice-BlankEN" \
    --onnx_path "$PRETRAINED_DIR" \
    --model "$MODEL" \
    --checkpoint "$CKPT" \
    --model_dir "$MODEL_DIR" \
    --tensorboard_dir "$CHECKPOINT_DIR/tensorboard/$MODEL" \
    --ddp.dist_backend nccl \
    --num_workers 4 \
    --prefetch 100 \
    --pin_memory \
    --use_amp
