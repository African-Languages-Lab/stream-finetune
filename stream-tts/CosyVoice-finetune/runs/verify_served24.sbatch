#!/bin/bash
#SBATCH --job-name=verify24
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/verify24_%j.out
#SBATCH --error=logs/verify24_%j.err
source /leonardo_work/AIH4A_udutech/cosyvoice_venv/bin/activate
module load cuda/12.2
cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice-finetune
python3 scripts/try_new_models.py \
  --refs voice_selection/served24_refs.json \
  --out  voice_selection/served24_results.json \
  --wavdir voice_selection/served24_gen \
  --workdir /leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/verify24 \
  --langs hausa twi igbo ewe berber umbundu amharic arabic fula luganda lingala malagasy sepedi chichewa oromo somali sesotho swahili tigrinya tswana tsonga venda xhosa zulu \
  --codes ha-NG tw-GH ig-NG ee-GH ber-MA umb-AO am-ET  ar-AR ff-SN lg-UG   ln-CD   mg-MG    nso-ZA ny-MW    or-KE so-SO  st-ZA   sw-KE   ti-ER    tn-BW  ts-ZA  ve-ZA xh-ZA zu-ZA
