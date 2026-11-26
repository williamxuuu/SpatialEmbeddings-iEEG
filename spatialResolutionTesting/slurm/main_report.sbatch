#!/bin/bash
#SBATCH --job-name=main_report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
# Usage: sbatch slurm/main_report.sbatch xsess|xsubj
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/activate_env.sh
KIND="$1"
python scripts/main_report.py "$KIND" "results/main_$KIND" --out "reports/main_$KIND.md"
