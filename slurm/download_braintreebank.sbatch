#!/bin/bash
# Download the 12 Neuroprobe Lite recordings of Brain Treebank (CPU-only job).
#   sbatch --account=... --qos=... slurm/download_braintreebank.sbatch
#
# Set ROOT_DIR_BRAINTREEBANK in slurm/cluster.env to a path ENDING IN
# /braintreebank, e.g. /storage/scratch1/<you>/braintreebank. Neuroprobe's
# script creates that folder (and a temporary braintreebank_zip/ next to it)
# in the directory it runs from, so this job runs it in the parent folder.
#
# Size: 75 GB zipped. Zips are deleted one by one as they are extracted, so
# peak disk use is roughly zipped + unzipped (~200 GB, a guess).
# If compute nodes have no internet, run the same commands on a login or
# data-transfer node instead.
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
ROOT="${ROOT_DIR_BRAINTREEBANK%/}"
if [ "$(basename "$ROOT")" != "braintreebank" ]; then
  echo "ROOT_DIR_BRAINTREEBANK must end in /braintreebank (got $ROOT)"; exit 1
fi
PARENT="$(dirname "$ROOT")"
mkdir -p "$PARENT" third_party
NP_COMMIT=b282acd497134bf0454cd36e14fb03bf0fa91c42   # same commit as pyproject.toml
if [ ! -d third_party/neuroprobe ]; then
  git clone https://github.com/azaho/neuroprobe third_party/neuroprobe
fi
git -C third_party/neuroprobe checkout -q "$NP_COMMIT"
cd "$PARENT"
python "${SLURM_SUBMIT_DIR}/third_party/neuroprobe/braintreebank_download_extract.py" --lite
# The script prints "Failed to extract" and carries on, so check the result.
missing=0
for f in sub_1_trial001 sub_1_trial002 sub_2_trial000 sub_2_trial004 sub_3_trial000 sub_3_trial001 \
         sub_4_trial000 sub_4_trial001 sub_7_trial000 sub_7_trial001 sub_10_trial000 sub_10_trial001; do
  [ -f "$ROOT/$f.h5" ] || { echo "MISSING $ROOT/$f.h5"; missing=1; }
done
[ "$missing" = 0 ] && echo "All 12 Lite recordings present in $ROOT" || exit 1
