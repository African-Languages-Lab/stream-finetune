#!/bin/bash
#SBATCH --job-name=cosyft-inference-igbo-clone-retry
#SBATCH --account=aih4a_udutech
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/inference_igbo_drive/clone_retry_%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/inference_igbo_drive/clone_retry_%j.out

source /leonardo_work/AIH4A_udutech/cosyvoice_venv/bin/activate
module load cuda/12.2
cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune/scripts
python3 run_inference_igbo_clone_retry.py
