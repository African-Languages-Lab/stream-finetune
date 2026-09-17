#!/bin/bash
cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2
N=voice_selection/new5; W=voice_selection/new5_wavs
printf '%s\n' swati bambara kanuri fon krio \
 | xargs -P 5 -I{} sh -c "python3 scripts/cands2.py --out $N/{}.json --wavdir $W --langs {} --want 3 > $N/{}.log 2>&1"
touch $N/_DONE
