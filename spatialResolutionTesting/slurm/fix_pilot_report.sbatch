#!/bin/bash
#SBATCH --job-name=fix_pilot_report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/activate_env.sh
python scripts/fix_pilot_report.py results/fix_pilot > reports/fix_pilot.md && cat reports/fix_pilot.md
