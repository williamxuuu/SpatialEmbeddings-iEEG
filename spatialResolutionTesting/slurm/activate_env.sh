#!/bin/bash
# Sourced by every job and helper script: loads modules, activates the Python
# env, exports paths. Usage: source slurm/activate_env.sh
_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$_repo/slurm/cluster.env" ]; then
  set -a; source "$_repo/slurm/cluster.env"; set +a
fi
if type module >/dev/null 2>&1 && [ -n "${LOCEMB_PYTHON_MODULE:-}" ]; then
  module load "$LOCEMB_PYTHON_MODULE" >/dev/null 2>&1 || echo "WARN: could not load module $LOCEMB_PYTHON_MODULE"
fi
_env="${LOCEMB_ENV_DIR:-$_repo/.venv}"
if [ -f "$_env/bin/activate" ]; then
  set +u; source "$_env/bin/activate"; set -u 2>/dev/null || true
else
  echo "WARN: no env at $_env (run: bash slurm/setup_env.sh)"
fi
export LOCEMB_ATLAS_DIR="${LOCEMB_ATLAS_DIR:-$_repo/nilearn_data}"
export PYTHONUNBUFFERED=1
[ -n "${ROOT_DIR_BRAINTREEBANK:-}" ] && export ROOT_DIR_BRAINTREEBANK
[ -n "${LOCEMB_CACHE_DIR:-}" ] && export LOCEMB_CACHE_DIR
unset _repo _env
