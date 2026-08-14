#!/bin/bash
# Runs CosyVoice's own make_parquet_list.py against a regime's Kaldi dir (built by
# 04_build_regime_kaldi.py, or the global dir directly for the combined regime) for both
# splits, producing parquet shards + train.data.list / dev.data.list for training.
set -eu
REGIME=$1  # e.g. individual_rw-RW, cluster_afroasiatic, combined

COSYVOICE_ROOT=/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice
if [ "$REGIME" = "combined" ]; then
    KALDI_ROOT=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/kaldi_data/global
else
    KALDI_ROOT=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/kaldi_data/$REGIME
fi
PARQUET_ROOT=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/parquet/$REGIME

for split in train dev; do
    src="$KALDI_ROOT/$split"
    des="$PARQUET_ROOT/$split"
    mkdir -p "$des"
    if [ -f "$des/data.list" ]; then
        echo "=== $REGIME/$split parquet already built, skipping ==="
        continue
    fi
    python3 "$COSYVOICE_ROOT/tools/make_parquet_list.py" \
        --num_utts_per_parquet 1000 \
        --num_processes 10 \
        --src_dir "$src" \
        --des_dir "$des"
done

# train.py wants a single train_data / cv_data file (list of parquet shard paths).
cp "$PARQUET_ROOT/train/data.list" "$PARQUET_ROOT/train.data.list"
cp "$PARQUET_ROOT/dev/data.list" "$PARQUET_ROOT/dev.data.list"
echo "=== $REGIME parquet ready at $PARQUET_ROOT ==="
