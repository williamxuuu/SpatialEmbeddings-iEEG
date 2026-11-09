#!/bin/bash
# Run one config by hand, e.g. a GPU smoke test:
#   sbatch slurm/single.sbatch configs/smoke.yaml spatial.type=region spatial.atlas=grid20
# Add your account/QOS on the sbatch line, e.g. sbatch --account=gts-xxx --qos=embers slurm/single.sbatch ...
#SBATCH --job-name=locemb-one
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/activate_env.sh
CONFIG="$1"; shift
python -m locemb.train --config "$CONFIG" "$@"
