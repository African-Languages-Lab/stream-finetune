#!/bin/bash
# Build the omnilingual environment once. Compute nodes have no internet, so everything
# that needs downloading happens here, on a node that does.
#SBATCH --job-name=omni-setup
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --account=AIH4A_udutech
#SBATCH --output=/leonardo_scratch/large/userexternal/atsado00/omni_ft/logs/setup_%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/atsado00/omni_ft/logs/setup_%j.err
set -e
ENV=/leonardo_work/AIH4A_udutech/omni_venv
module load cuda/12.2 || true
python3 -m venv $ENV
source $ENV/bin/activate
pip install --upgrade pip
# fairseq2 needs a build matched to the torch/CUDA pair -- see the omnilingual README.
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install fairseq2==0.5.0 --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.6.0/cu124
pip install -e /leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-asr/omnilingual-asr
python -c "import fairseq2, omnilingual_asr; print('fairseq2', fairseq2.__version__)"
