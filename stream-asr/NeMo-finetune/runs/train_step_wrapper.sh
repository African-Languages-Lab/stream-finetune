#!/bin/bash
# Invoked by srun --exclusive -N1 inside run_cluster_train.sbatch / run_individual_train.sbatch.
# When multiple --exclusive -N1 steps run concurrently within one multi-node sbatch job,
# PyTorch Lightning's SLURMEnvironment derives MASTER_ADDR from the whole job's node list
# rather than this step's own node -- every concurrent step then tries to rendezvous on the
# same (wrong) node, which fails with an NCCL socket connection error. Since each step is
# confined to exactly one physical node, all 4 ranks on it independently compute the same
# correct MASTER_ADDR via their own hostname, sidestepping that ambiguity entirely.
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=29500

CONFIG_PATH="$1"
CONFIG_NAME="$2"

cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-asr/NeMo
exec python examples/asr/speech_to_text_finetune.py \
    --config-path="$CONFIG_PATH" \
    --config-name="$CONFIG_NAME"
