#!/bin/bash
#SBATCH --job-name=cosyft-regime
#SBATCH --account=aih4a_udutech
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=23:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --signal=B:TERM@180
#SBATCH --output=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/logs/regime_%x_%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/logs/regime_%x_%j.out

# Trains one regime's llm -> flow -> hifigan in sequence, one "cycle" (up to the job's wall
# time) per stage before moving to the next. CosyVoice's training loop is epoch-based with no
# built-in step cap (see cosyvoice/utils/executor.py), so we bound effort externally via a
# per-tier target cycle count per stage instead, tracked in a small JSON state file that
# survives resubmission. normal QOS has no account-wide TRES cap (unlike boost_qos_lprod,
# which the Nemotron ASR combined job already saturates at 8 nodes/32 GPUs) but caps wall
# time at the partition's 24h -- we catch the pre-timeout signal, advance/persist state, and
# resubmit ourselves, same pattern proven on the ASR pipeline.
#
# Usage: sbatch run_regime_train.sbatch <regime_dir_name> <kind: individual|cluster|combined> <cycles_per_stage>
# regime_dir_name matches the Kaldi/parquet dir name, e.g. individual_rw-RW, cluster_afroasiatic, combined

REGIME=$1
KIND=$2
CYCLES_PER_STAGE=$3

RUNS_DIR=/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune/runs
KALDI_DIR=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/kaldi_data/$REGIME
PARQUET_DIR=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/parquet/$REGIME
CHECKPOINT_DIR=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/$KIND/$REGIME
STATE_FILE=$CHECKPOINT_DIR/stage.state.json
DONE_MARKER=$CHECKPOINT_DIR/$REGIME.done
mkdir -p "$CHECKPOINT_DIR"

if [ -f "$DONE_MARKER" ]; then
    echo "=== $REGIME already done, exiting ==="
    exit 0
fi

source /leonardo_work/AIH4A_udutech/cosyvoice_venv/bin/activate
module load cuda/12.2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAGES=(llm flow hifigan)

read_state() {
    python3 -c "
import json, os
p = '$STATE_FILE'
if os.path.exists(p):
    s = json.load(open(p))
else:
    s = {'stage_idx': 0, 'cycles_done': 0}
print(s['stage_idx'], s['cycles_done'])
"
}

write_state() {
    python3 -c "
import json
json.dump({'stage_idx': $1, 'cycles_done': $2}, open('$STATE_FILE', 'w'))
"
}

resubmit_or_finish() {
    local stage_idx=$1
    local cycles_done=$2
    if [ "$cycles_done" -ge "$CYCLES_PER_STAGE" ]; then
        stage_idx=$((stage_idx + 1))
        cycles_done=0
    fi
    if [ "$stage_idx" -ge 3 ]; then
        echo "=== $REGIME: all stages reached their cycle target, marking done ==="
        touch "$DONE_MARKER"
        exit 0
    fi
    write_state "$stage_idx" "$cycles_done"
    echo "=== $REGIME: resubmitting at stage=${STAGES[$stage_idx]} cycle=$cycles_done/$CYCLES_PER_STAGE ==="
    sbatch --job-name="cosyft-$KIND-$REGIME" "$RUNS_DIR/run_regime_train.sbatch" "$REGIME" "$KIND" "$CYCLES_PER_STAGE"
    exit 0
}

read STAGE_IDX CYCLES_DONE < <(read_state)
MODEL=${STAGES[$STAGE_IDX]}
echo "=== $REGIME: stage=$MODEL (idx=$STAGE_IDX) cycle=$CYCLES_DONE/$CYCLES_PER_STAGE ==="

trap 'resubmit_or_finish $STAGE_IDX $((CYCLES_DONE + 1))' TERM

bash "$RUNS_DIR/train_one_stage.sh" "$REGIME" "$MODEL" 4 "$CHECKPOINT_DIR" "$PARQUET_DIR" &
TRAIN_PID=$!
wait $TRAIN_PID
TRAIN_EXIT=$?

# Reached here without the TERM trap firing: the stage's data loader ran out (epoch-based
# training with no step cap can still finish an epoch faster than the wall-time limit on
# small regimes) rather than being cut off by wall time. Count it as a completed cycle either
# way and move on the same as the timeout path.
echo "=== $REGIME/$MODEL exited on its own (code $TRAIN_EXIT), counting as one cycle ==="
resubmit_or_finish "$STAGE_IDX" $((CYCLES_DONE + 1))
