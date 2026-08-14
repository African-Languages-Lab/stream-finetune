#!/bin/bash
#SBATCH --job-name=nemotron-tokenizer
#SBATCH --account=aih4a_udutech
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/logs/tokenizer-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/logs/tokenizer-%j.out

source /leonardo_work/AIH4A_udutech/nemo_venv/bin/activate
cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-asr/NeMo

python scripts/tokenizers/process_asr_text_tokenizer.py \
    --data_file=/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/tokenizer/corpus.txt \
    --data_root=/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/tokenizer \
    --vocab_size=32000 \
    --tokenizer=spe \
    --spe_type=unigram \
    --spe_character_coverage=1.0 \
    --spe_byte_fallback \
    --spe_remove_extra_whitespaces \
    --log
