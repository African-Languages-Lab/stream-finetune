#!/bin/bash
#SBATCH --job-name=cosyft-smoke-train
#SBATCH --account=aih4a_udutech
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/logs/smoke_train_%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/logs/smoke_train_%j.out

source /leonardo_work/AIH4A_udutech/cosyvoice_venv/bin/activate
module load cuda/12.2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUNS_DIR=/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune/runs
CHECKPOINT_DIR=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/smoke
PARQUET_DIR=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/parquet/smoke

# use timeout instead of the full TERM-trap resubmit flow -- this is a one-shot validation,
# not meant to actually train to completion or resubmit itself. All three stages (including
# hifigan, which needed the hift.pt checkpoint-naming fix) get a slice of the 20 min debug
# budget, enough for each to log at least a few real steps.
for MODEL in llm flow hifigan; do
    echo "########## SMOKE STAGE: $MODEL ##########"
    timeout 240 bash "$RUNS_DIR/train_one_stage.sh" smoke "$MODEL" 1 "$CHECKPOINT_DIR" "$PARQUET_DIR" || true
done
