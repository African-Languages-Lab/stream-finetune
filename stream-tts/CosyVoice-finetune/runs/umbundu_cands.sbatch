#!/bin/bash
#SBATCH --job-name=umbcand
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --account=AIH4A_udutech
#SBATCH --output=/leonardo_scratch/large/userexternal/atsado00/fixrefs/logs/umb_%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/atsado00/fixrefs/logs/umb_%j.err
module load cuda/12.2
source /leonardo_work/AIH4A_udutech/cosyvoice_venv/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2
cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune
SC=/leonardo_scratch/large/userexternal/atsado00/fixrefs
python3 scripts/cands2.py --out $SC/json/umbundu.json --wavdir $SC/wavs --langs umbundu --want 4
touch $SC/json/_UMB_DONE
