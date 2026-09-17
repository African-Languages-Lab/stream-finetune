#!/bin/bash
cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2
R=voice_selection/redo; W=voice_selection/redo_wavs
( python3 scripts/deep_search.py --lang igbo    --genders male female --out $R/igbo.json    --wavdir $W > $R/igbo.log 2>&1 ) &
( python3 scripts/deep_search.py --lang lingala --genders male        --out $R/lingala.json --wavdir $W > $R/lingala.log 2>&1 ) &
( python3 scripts/deep_search.py --lang swahili --genders female      --out $R/swahili.json --wavdir $W > $R/swahili.log 2>&1 ) &
( python3 scripts/deep_search.py --lang twi     --genders female      --out $R/twi.json     --wavdir $W > $R/twi.log 2>&1 ) &
wait
touch $R/_DONE
