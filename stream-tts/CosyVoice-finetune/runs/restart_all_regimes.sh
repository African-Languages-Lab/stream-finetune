#!/bin/bash
# Full restart of every CosyVoice3 regime under the new multi-node + step-target-with-CV-early-
# stopping scheme (see run_regime_train.sbatch, check_stage_progress.py). Old checkpoints were
# archived, not deleted, to checkpoints_archived_20260818/ -- every regime here starts fresh
# from the pretrained release checkpoint. en-ZA is skipped: no parquet data built yet.
set -eu
RUNS_DIR=/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune/runs

CLUSTERS="afroasiatic east_africa_bantu southern_central_bantu west_africa_niger_congo other_languages"
LANGS="af-ZA am-ET ar-AR bem-ZM ber-MA bm-ML ee-GH en-GH en-NG en-UG ff-SN fon-BJ ha-NG ig-NG ki-KE kri-SL kr-NG lg-UG ln-CD mg-MG nd-ZW nso-ZA ny-MW or-KE rw-RW sn-ZW so-SO ss-SZ st-ZA sw-KE ti-ER tn-BW ts-ZA tw-GH umb-AO ve-ZA wo-SN xh-ZA yo-NG zu-ZA"
# en-ZA excluded -- no parquet data built

echo "=== combined: 16 nodes ==="
sbatch --nodes=16 --ntasks=16 --ntasks-per-node=1 --gres=gpu:4 --cpus-per-task=32 \
    --job-name="cosyft-combined-combined" "$RUNS_DIR/run_regime_train.sbatch" "combined" "combined" 16

for c in $CLUSTERS; do
    echo "=== cluster: $c, 8 nodes ==="
    sbatch --nodes=8 --ntasks=8 --ntasks-per-node=1 --gres=gpu:4 --cpus-per-task=32 \
        --job-name="cosyft-cluster-$c" "$RUNS_DIR/run_regime_train.sbatch" "cluster_$c" "cluster" 8
done

for l in $LANGS; do
    echo "=== individual: $l, 4 nodes ==="
    sbatch --nodes=4 --ntasks=4 --ntasks-per-node=1 --gres=gpu:4 --cpus-per-task=32 \
        --job-name="cosyft-individual-$l" "$RUNS_DIR/run_regime_train.sbatch" "individual_$l" "individual" 4
done

echo "=== all regimes submitted (39 individual + 5 cluster + 1 combined = 45 jobs, 212 nodes) ==="
