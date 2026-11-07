#!/usr/bin/env bash
# Runner for the RGF experiments inside the faithful WSL environment.
#
#   wsl -e bash scripts/run_wsl.sh <python args...>
#
# e.g. wsl -e bash scripts/run_wsl.sh scripts/eval_predictor_baseline.py --graphs 500
#
# GPU is pinned to device 1 by default (the user's daily-use card stays on
# device 0); override with CUDA_VISIBLE_DEVICES, which is honored inside WSL.
#
# The repository root is derived from this script's own location, so the same
# file works in any checkout (no hard-coded absolute path).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-difusco}"
CONDA="${HOME}/miniconda3"
PY="${CONDA}/envs/${ENV_NAME}/bin/python"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export LD_LIBRARY_PATH="${CONDA}/envs/${ENV_NAME}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
# the third-party predictors resolve their data/checkpoint roots relative to the
# repository, so run from the experiment root
cd "${REPO_ROOT}"

exec "${PY}" "$@"
