#!/bin/bash
#SBATCH --job-name=check_smoke_main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
# Gate: exits non-zero unless the concurrent smoke test passed (scripts/check_smoke_main.py).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/activate_env.sh
python scripts/check_smoke_main.py results/smoke_main
