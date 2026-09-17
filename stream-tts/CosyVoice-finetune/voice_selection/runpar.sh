#!/bin/bash
cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2
P=voice_selection/par; W=voice_selection/wavs
printf '%s\n' xhosa sepedi sesotho somali fula twi amharic oromo malagasy arabic igbo \
  lingala swahili zulu hausa berber ewe umbundu chichewa tigrinya luganda tsonga tswana venda \
| xargs -P 12 -I{} sh -c "python3 scripts/cands2.py --out $P/{}.json --wavdir $W --langs {} > $P/{}.log 2>&1"
touch $P/_COMPLETE
