#!/bin/bash
#SBATCH --job-name=smoke_valperson
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out
# Smoke test: validation person (data.val_subjects=auto) + region-code dropout, 3 short epochs.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/activate_env.sh
_local="${TMPDIR:-/tmp}/locemb_cache_${SLURM_JOB_ID}"
mkdir -p "$_local"; trap 'rm -rf "$_local"' EXIT
rsync -a "$LOCEMB_CACHE_DIR/" "$_local/" && export LOCEMB_CACHE_DIR="$_local"
python -m locemb.train --config configs/neuroprobe_v2.yaml exp_name=smoke_valperson run_name=ho01_s400hard_drop05 \
  results_dir=results force=true model.name=transformer data.split=cross_subject data.eval_name=speech data.electrodes=all \
  "data.subject_trials=[[1,2],[2,4],[3,1],[4,1],[7,1],[10,1]]" "data.test_subjects=[1]" data.val_subjects=auto \
  model.frontend=stft preprocess.zscore=session preprocess.reref=car model.readout=cls \
  spatial.type=region spatial.atlas=schaefer400 spatial.mode=hard spatial.code_dropout=0.5 \
  train.epochs=3 train.min_epochs=3 train.select=best train.max_batches=300
