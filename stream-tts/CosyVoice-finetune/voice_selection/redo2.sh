#!/bin/bash
cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2
R=voice_selection/redo2; W=voice_selection/redo_wavs
( python3 scripts/deep_search.py --lang twi     --genders male --out $R/twi.json     --wavdir $W --want 5 > $R/twi.log 2>&1 ) &
( python3 scripts/deep_search.py --lang luganda --genders male --out $R/luganda.json --wavdir $W --want 5 > $R/luganda.log 2>&1 ) &
wait
touch $R/_DONE
