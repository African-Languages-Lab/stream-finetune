#!/bin/bash
# One-time launcher: builds each regime's Kaldi dir + parquet shards (CPU-only, cheap) then
# submits its training job. Run from a login node after the global feature extraction
# (03_merge_features.py) has completed -- regime construction slices that global pool, it
# doesn't re-extract anything.
#
# Cycle targets (each cycle = up to one 24h wall-time slice per stage, see
# run_regime_train.sbatch) are a starting point for this research fine-tune, scaled loosely
# by regime data volume: individual=1, cluster=2, combined=3. Bump the target and resubmit
# (or just re-run this with a higher CYCLES value for a given regime) to train further.
set -eu
SCRIPTS_DIR=/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune/scripts
RUNS_DIR=/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune/runs

CLUSTERS="afroasiatic east_africa_bantu southern_central_bantu west_africa_niger_congo other_languages"
LANGS="af-ZA am-ET ar-AR bem-ZM ber-MA bm-ML ee-GH en-GH en-NG en-UG en-ZA ff-SN fon-BJ ha-NG ig-NG ki-KE kri-SL kr-NG lg-UG ln-CD mg-MG nd-ZW nso-ZA ny-MW or-KE rw-RW sn-ZW so-SO ss-SZ st-ZA sw-KE ti-ER tn-BW ts-ZA tw-GH umb-AO ve-ZA wo-SN xh-ZA yo-NG zu-ZA"

echo "=== combined ==="
bash "$SCRIPTS_DIR/05_make_regime_parquet.sh" combined
sbatch --job-name=cosyft-combined "$RUNS_DIR/run_regime_train.sbatch" combined combined 3

for c in $CLUSTERS; do
    echo "=== cluster: $c ==="
    python3 "$SCRIPTS_DIR/04_build_regime_kaldi.py" --regime "$c" --kind cluster
    bash "$SCRIPTS_DIR/05_make_regime_parquet.sh" "cluster_$c"
    sbatch --job-name="cosyft-cl-$c" "$RUNS_DIR/run_regime_train.sbatch" "cluster_$c" cluster 2
done

for l in $LANGS; do
    echo "=== individual: $l ==="
    python3 "$SCRIPTS_DIR/04_build_regime_kaldi.py" --regime "$l" --kind individual
    bash "$SCRIPTS_DIR/05_make_regime_parquet.sh" "individual_$l"
    sbatch --job-name="cosyft-ind-$l" "$RUNS_DIR/run_regime_train.sbatch" "individual_$l" individual 1
done

echo "=== all 47 regimes submitted ==="
