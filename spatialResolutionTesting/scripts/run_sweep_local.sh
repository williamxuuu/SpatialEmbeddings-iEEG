#!/bin/bash
# Run a sweep one run at a time WITHOUT Slurm (laptop / interactive GPU session).
#   bash scripts/run_sweep_local.sh configs/sweeps/smoke.yaml [extra key=value overrides]
set -euo pipefail
SWEEP="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; shift   # works from any folder
cd "$(dirname "$0")/.."
read -r N FILE < <(python -m locemb.sweep "$SWEEP")
for ((i=0; i<N; i++)); do
  echo "=== run $((i+1))/$N ==="
  python -m locemb.train --sweep-file "$FILE" --index "$i" "$@"
done
