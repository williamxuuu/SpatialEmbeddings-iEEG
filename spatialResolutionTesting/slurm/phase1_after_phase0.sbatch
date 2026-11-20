#!/bin/bash
# Runs once Phase 0 (both arrays) has ended, via
#   sbatch --dependency=afterany:<phase0_recipe job>:<phase0_xsubj job> slurm/phase1_after_phase0.sbatch
# CPU only. Applies the pre-registered gate (scripts/phase0_gate.py). On GO it writes
# the picked recipe into configs/neuroprobe_v2.yaml, commits + pushes that and the
# report, and submits Phase 1 (full_xsess + full_xsubj). On NO-GO it submits nothing.
#SBATCH --job-name=phase0_gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/activate_env.sh
echo "[gate] start=$(date)"
out=$(python scripts/phase0_gate.py --apply); rc=$?
echo "$out"
last=$(echo "$out" | tail -1)
cd .. 
git add spatialResolutionTesting/reports/phase0_gate.md spatialResolutionTesting/reports/phase0_report.txt \
        spatialResolutionTesting/reports/phase0_pick.json spatialResolutionTesting/configs/neuroprobe_v2.yaml 2>/dev/null
git commit -q -m "Phase 0 gate: ${last}

$(cat spatialResolutionTesting/reports/phase0_gate.md)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q || echo "[gate] commit/push failed (results are still on disk)"
cd spatialResolutionTesting
if [ $rc -ne 0 ] || [[ "$last" != GO* ]]; then
  echo "[gate] NO-GO: Phase 1 not submitted. end=$(date)"; exit 0
fi
MIN=${last#GO }
H=$(( MIN / 60 )); M=$(( MIN % 60 ))
TIME=$(printf "%02d:%02d:00" $H $M)
# Don't let this job's own Slurm settings leak into the Phase 1 submissions.
for v in $(compgen -e | grep '^SLURM_\|^SBATCH_'); do unset "$v"; done
echo "[gate] GO: submitting Phase 1, 3 runs per task, --time=$TIME"
LOCEMB_PER_TASK=3 bash slurm/submit_sweep.sh configs/sweeps/full/full_xsess.yaml 8 --time=$TIME
LOCEMB_PER_TASK=3 bash slurm/submit_sweep.sh configs/sweeps/full/full_xsubj.yaml 12 --time=$TIME
echo "[gate] end=$(date)"
