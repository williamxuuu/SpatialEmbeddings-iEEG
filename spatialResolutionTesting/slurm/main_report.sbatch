#!/bin/bash
#SBATCH --job-name=main_report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
# Usage: sbatch slurm/main_report.sbatch xsess|xsubj|xsoft|xsubj2|xheld
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/activate_env.sh
KIND="$1"
if [ "$KIND" = xsoft ]; then
  python scripts/main_report.py xsoft results/main_soft_xsess results/main_xsess --out reports/main_soft.md
elif [ "$KIND" = xsubj2 ]; then
  python scripts/main_report.py xsubj2 results/main_xsubj results/main_xsubj_ext --out reports/main_xsubj_ext.md
elif [ "$KIND" = xheld ]; then
  python scripts/main_report.py xheld results/heldout_shafts --out reports/heldout_shafts.md
else
  python scripts/main_report.py "$KIND" "results/main_$KIND" --out "reports/main_$KIND.md"
fi
