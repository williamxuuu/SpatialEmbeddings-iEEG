#!/bin/bash
# Download Brain Treebank (CPU-only job; set ROOT_DIR_BRAINTREEBANK in cluster.env first).
#   sbatch --account=... --qos=... slurm/download_braintreebank.sbatch
# VERIFY: check the download script's options in the Neuroprobe repo first;
# some clusters prefer big downloads on a data-transfer node instead.
#SBATCH --job-name=btb-download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/activate_env.sh
: "${ROOT_DIR_BRAINTREEBANK:?set ROOT_DIR_BRAINTREEBANK in slurm/cluster.env}"
mkdir -p "$ROOT_DIR_BRAINTREEBANK" third_party
[ -d third_party/neuroprobe ] || git clone https://github.com/azaho/neuroprobe third_party/neuroprobe
cd "$ROOT_DIR_BRAINTREEBANK"
python "${SLURM_SUBMIT_DIR}/third_party/neuroprobe/braintreebank_download_extract.py" --lite
