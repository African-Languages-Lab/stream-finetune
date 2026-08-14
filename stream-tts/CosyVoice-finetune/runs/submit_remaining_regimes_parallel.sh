#!/bin/bash
# Submits the 5 cluster + 41 individual regimes as SEPARATE parallel build_and_launch_regime
# jobs (combined is left alone -- already ~48% through its parquet build in the original
# sequential orchestrator job, job 52293731, no point restarting that sunk work). Each job
# here does its own data-prep then submits its own training job, so all 46 regimes' data-prep
# proceeds concurrently across as many nodes as SLURM will schedule, instead of waiting in
# a single sequential queue.
set -eu
RUNS_DIR=/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune/runs

CLUSTERS="afroasiatic east_africa_bantu southern_central_bantu west_africa_niger_congo other_languages"
LANGS="af-ZA am-ET ar-AR bem-ZM ber-MA bm-ML ee-GH en-GH en-NG en-UG en-ZA ff-SN fon-BJ ha-NG ig-NG ki-KE kri-SL kr-NG lg-UG ln-CD mg-MG nd-ZW nso-ZA ny-MW or-KE rw-RW sn-ZW so-SO ss-SZ st-ZA sw-KE ti-ER tn-BW ts-ZA tw-GH umb-AO ve-ZA wo-SN xh-ZA yo-NG zu-ZA"

for c in $CLUSTERS; do
    sbatch --job-name="cosyft-build-cl-$c" "$RUNS_DIR/build_and_launch_regime.sbatch" "$c" cluster 2
done

for l in $LANGS; do
    sbatch --job-name="cosyft-build-ind-$l" "$RUNS_DIR/build_and_launch_regime.sbatch" "$l" individual 1
done

echo "=== submitted build jobs for 5 clusters + 41 individual languages ==="
