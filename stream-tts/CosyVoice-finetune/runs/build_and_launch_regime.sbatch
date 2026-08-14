#!/bin/bash
#SBATCH --job-name=cosyft-build
#SBATCH --account=aih4a_udutech
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/logs/build_%x_%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/logs/build_%x_%j.out

# One job per regime for data-prep too (not just training): builds this regime's Kaldi dir +
# parquet shards, then submits its own training job. Submitting all regimes' build jobs at
# once lets SLURM run the CPU/IO-bound packaging step for many regimes concurrently across
# nodes, instead of the earlier single-orchestrator design that built all 47 regimes'
# data one at a time before most training jobs ever got submitted.
#
# Usage: sbatch --job-name=cosyft-build-<name> build_and_launch_regime.sbatch <regime> <kind> <cycles>
#
# set -e matters here: without it, a failed parquet build (confirmed: happened for real --
# other_languages hit a KeyError deep in make_parquet_list.py's multiprocessing worker) let
# this script fall through to submitting a training job anyway, against incomplete/broken
# data, which then just failed a second time downstream instead of failing loudly here.
set -e
REGIME=$1
KIND=$2
CYCLES=$3

SCRIPTS_DIR=/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune/scripts
RUNS_DIR=/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune/runs

source /leonardo_work/AIH4A_udutech/cosyvoice_venv/bin/activate

if [ "$KIND" != "combined" ]; then
    python3 "$SCRIPTS_DIR/04_build_regime_kaldi.py" --regime "$REGIME" --kind "$KIND"
fi
PARQUET_NAME="$REGIME"
[ "$KIND" = "cluster" ] && PARQUET_NAME="cluster_$REGIME"
[ "$KIND" = "individual" ] && PARQUET_NAME="individual_$REGIME"

bash "$SCRIPTS_DIR/05_make_regime_parquet.sh" "$PARQUET_NAME"
sbatch --job-name="cosyft-$KIND-$REGIME" "$RUNS_DIR/run_regime_train.sbatch" "$PARQUET_NAME" "$KIND" "$CYCLES"
