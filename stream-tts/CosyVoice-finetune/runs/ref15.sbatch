#!/bin/bash
#SBATCH --job-name=ref15
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --account=AIH4A_udutech
#SBATCH --output=/leonardo_scratch/large/userexternal/atsado00/ref15/logs/%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/atsado00/ref15/logs/%j.err
module load cuda/12.2
source /leonardo_work/AIH4A_udutech/cosyvoice_venv/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2
cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune
SC=/leonardo_scratch/large/userexternal/atsado00/ref15
printf '%s\n' hausa twi igbo ewe berber umbundu amharic arabic fula luganda lingala malagasy \
  sepedi chichewa oromo somali sesotho swahili tigrinya tswana tsonga venda xhosa zulu \
 | xargs -P 12 -I{} sh -c "python3 scripts/cands15.py --out $SC/json/{}.json --wavdir $SC/wavs --langs {} --want 3 > $SC/logs/{}.log 2>&1"
touch $SC/json/_DONE
