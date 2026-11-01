#!/bin/bash
# Run a sweep one run at a time WITHOUT Slurm (laptop / interactive GPU session).
#   bash scripts/run_sweep_local.sh configs/sweeps/smoke.yaml [extra key=value overrides]
set -euo pipefail
cd "$(dirname "$0")/.."
SWEEP="$1"; shift
read -r N FILE < <(python -m locemb.sweep "$SWEEP")
for ((i=0; i<N; i++)); do
  echo "=== run $((i+1))/$N ==="
  python -m locemb.train --sweep-file "$FILE" --index "$i" "$@"
done
