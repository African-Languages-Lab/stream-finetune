#!/bin/bash
cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2
R=voice_selection/relen; W=voice_selection/relen_wavs
printf '%s\n' hausa arabic berber chichewa oromo sepedi somali tswana twi fula lingala amharic \
 | xargs -P 12 -I{} sh -c "python3 scripts/cands2.py --out $R/{}.json --wavdir $W --langs {} --want 3 > $R/{}.log 2>&1"
touch $R/_DONE
