#!/bin/bash
# Trains one CosyVoice3 component (llm, flow, or hifigan) for one regime's data, resuming
# from that stage's latest self-produced checkpoint if one exists, otherwise initializing
# from the pretrained release checkpoint. Meant to be invoked once per SLURM job (a wall-time
# signal ends the torchrun process cleanly; the caller decides whether to resubmit).
#
# Usage: train_one_stage.sh <regime_name> <model: llm|flow|hifigan> <nodes> <checkpoint_dir> <parquet_dir>
# Multi-node: SLURM allocates $NODES nodes to the job; srun launches one torchrun per node
# (--ntasks-per-node=1), each spawning GPUS_PER_NODE local worker processes -- this only works
# because run_regime_train.sbatch requests the matching --nodes/--ntasks at submission time.
set -eu

REGIME=$1
MODEL=$2
NODES=$3
CHECKPOINT_DIR=$4   # e.g. /leonardo_scratch/.../checkpoints/individual/rw-RW
PARQUET_DIR=$5      # e.g. /leonardo_scratch/.../parquet/individual_rw-RW

GPUS_PER_NODE=4
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
    hifigan)
        # Belt and braces: whatever calls this, the vocoder is not fine-tuned. Every serving
        # model pairs a fine-tuned llm+flow with the ORIGINAL pretrained hift.pt, so training
        # one produces a checkpoint nothing will ever load.
        echo "refusing to train hifigan: the vocoder is never fine-tuned, use the pretrained hift.pt" >&2
        exit 0 ;;
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

# Opt-in wandb logging (see cosyvoice/utils/train_utils.py::init_summarywriter) -- offline by
# default since compute nodes here aren't guaranteed outbound internet; wandb sync the run
# dir from a login node, or export WANDB_MODE=online before submitting if that's confirmed fine.
export WANDB_PROJECT="${WANDB_PROJECT:-cosyvoice3-african}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-$(basename "$CHECKPOINT_DIR")-$MODEL}"

# Confirmed via NCCL_DEBUG=INFO,NET earlier: genuinely using InfiniBand with GPU Direct RDMA
# on all 4 HCAs ("Using network IB") -- not a network misconfiguration. The real multi-node
# speed picture (verified via epochs/hour, not misleading raw steps/hour) is ~4.2x faster than
# single-node, close to ideal linear scaling. Root-caused, diagnostic logging removed.

CV_DATA="$PARQUET_DIR/dev.data.list"
if [ "$MODEL" = "hifigan" ]; then
    # cosyvoice/utils/executor.py's train_one_epoc_gan runs a FULL cv() pass, unconditionally,
    # at the end of every single epoch (no config to disable or subsample it) -- confirmed this
    # was consuming 57+ minutes per pass for rw-RW while training itself advanced only ~100
    # steps in that same hour, because hifigan's per-batch cost (generator + discriminator +
    # mel-transform + F0-predictor on raw waveform, not compact tokens/mel) is far higher than
    # llm/flow's, and hifigan's epochs are short enough that this fires very often relative to
    # actual training progress. The dev split itself isn't oversized (27 of 1181 shards for
    # rw-RW, a normal ~2.3%) -- the fix is giving hifigan specifically a much smaller CV subset
    # for the loss-tracking/early-stop signal, since precise dev-set coverage doesn't matter
    # for that purpose the way it might for a final quality number.
    CV_DATA="$MODEL_DIR/cv_subset.data.list"
    head -n 2 "$PARQUET_DIR/dev.data.list" > "$CV_DATA"
    # Guard against an empty subset (a regime with <2 dev shards) -- an empty --cv_data would
    # hit the same empty-dataset crash already seen elsewhere (04_build_regime_kaldi.py's
    # ZeroDivisionError), so fall back to the full dev set rather than risk that.
    [ -s "$CV_DATA" ] || CV_DATA="$PARQUET_DIR/dev.data.list"
fi

# hifigan's data_pipeline_gan does real CPU-side work llm/flow's pipeline doesn't -- truncate
# plus compute_f0 (pitch extraction), on top of the compute_fbank both already pay for, none
# of it cached, recomputed from scratch every sample every epoch. 4 workers/GPU-process was
# leaving half the allocated CPU budget unused (--cpus-per-task=32 / 4 GPUs = 8 CPUs/process
# available). If those workers can't keep up, GPUs sit idle waiting on data -- this uses
# compute we already requested rather than asking for more nodes.
NUM_WORKERS=4
[ "$MODEL" = "hifigan" ] && NUM_WORKERS=8

# Multi-node rendezvous needs a real reachable address, not localhost -- the first node in
# this job's allocation acts as rendezvous host for all $NODES nodes' torchrun instances.
# Port is derived from the job ID (not fixed) so back-to-back resubmissions on nodes that
# were just released by a prior job don't ever collide on a lingering listener.
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

srun --ntasks="$NODES" --ntasks-per-node=1 \
    torchrun --nnodes="$NODES" --nproc_per_node="$GPUS_PER_NODE" \
    --rdzv_id="${SLURM_JOB_ID:-1}" --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    cosyvoice/bin/train.py \
    --train_engine torch_ddp \
    --config "$CONFIG" \
    --train_data "$PARQUET_DIR/train.data.list" \
    --cv_data "$CV_DATA" \
    --qwen_pretrain_path "$PRETRAINED_DIR/CosyVoice-BlankEN" \
    --onnx_path "$PRETRAINED_DIR" \
    --model "$MODEL" \
    --checkpoint "$CKPT" \
    --model_dir "$MODEL_DIR" \
    --tensorboard_dir "$CHECKPOINT_DIR/tensorboard/$MODEL" \
    --ddp.dist_backend nccl \
    --num_workers "$NUM_WORKERS" \
    --prefetch 100 \
    --pin_memory \
    --use_amp
