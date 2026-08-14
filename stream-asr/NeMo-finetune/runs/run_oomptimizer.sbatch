#!/bin/bash
#SBATCH --job-name=nemotron-oomptimizer
#SBATCH --account=aih4a_udutech
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/oomptimizer/oomptimizer-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/oomptimizer/oomptimizer-%j.out

# Empirically derives safe per-duration-bucket batch sizes for combined's exact model/config,
# replacing the 4-round manual guess-and-tighten cycle on batch_duration/max_duration/
# quadratic_duration/batch_size (see prep_final_config.py history) with NeMo's own tool built
# for exactly this problem. Buckets come from estimate_duration_bins.py run against combined's
# real 41-source train_blend (30 buckets spanning 3.3-30s, ~5.4% of utterances >30s discarded
# by the estimator's default filter -- matches the same population combined training draws from).
#
# Uses combined_oomptimizer_config.yaml (= combined_train_config.yaml with model.test_ds nulled
# out) instead of the real training config -- ModelPT.__init__ eagerly builds test_ds's
# dataloader whenever the key is present and non-null, and the real config's test_ds has
# manifest_filepath=null with no input_cfg, which throws IncompleteConfigError as soon as the
# model class is instantiated directly (confirmed: crashed the first attempt this way).
# speech_to_text_finetune.py must sidestep this some other way since real training runs fine;
# oomptimizer never touches test_ds either way, so nulling it out here is safe and scoped.
source /leonardo_work/AIH4A_udutech/nemo_venv/bin/activate
module load cuda/12.2
export NUMBA_CACHE_DIR="/leonardo_work/AIH4A_udutech/numba_cache"

cd /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-asr/NeMo

python scripts/speech_recognition/oomptimizer.py \
    --module-name nemo.collections.asr.models.EncDecRNNTBPEModelWithPrompt \
    --config-path /leonardo_scratch/large/userexternal/atsado00/nemotron_ft/configs/combined_oomptimizer_config.yaml \
    --optimizer-name adamw \
    --buckets 3.335,4.248,4.968,5.699,6.579,7.639,8.96,10.529,12.142,13.241,14.08,14.72,15.438,16.02,16.5,16.945,17.4,17.856,18.3,18.78,19.302,19.845,20.402,21.18,22.047,23.04,24.264,25.8,27.68,30.0 \
    --start-batch-size 32 \
    --device cuda:0
