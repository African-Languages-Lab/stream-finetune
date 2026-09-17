#!/bin/bash
#SBATCH --job-name=leakdet
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --account=AIH4A_udutech
#SBATCH --output=/leonardo_scratch/large/userexternal/atsado00/leakdetect/logs/%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/atsado00/leakdetect/logs/%j.err

module load cuda/12.2
source /leonardo_work/AIH4A_udutech/cosyvoice_venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SC=/leonardo_scratch/large/userexternal/atsado00/leakdetect
cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune
python scripts/leak_detect.py --out $SC/leak.json --wavdir $SC/wavs --workdir $SC/work
