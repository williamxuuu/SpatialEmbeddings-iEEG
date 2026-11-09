#!/bin/bash
# Submit an Experiment 2 sweep through spatialResolutionTesting's Slurm scripts
# (same code, Python env, data and filtered-recording cache).
#   bash softRegionTesting/submit.sh pilot_soft 8
#   bash softRegionTesting/submit.sh pilot_soft_xsubj 8 --dependency=afterok:<job id>
# Results go to softRegionTesting/results/<exp_name>/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NAME="${1:?usage: submit.sh <sweep name in configs/sweeps> [max_parallel] [extra sbatch args]}"; shift
exec bash "$HERE/../spatialResolutionTesting/slurm/submit_sweep.sh" "$HERE/configs/sweeps/$NAME.yaml" "$@"
