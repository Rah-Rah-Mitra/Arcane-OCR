#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Prevent leakage from previously sourced environments (e.g., hailo-apps setup_env.sh)
unset PYTHONPATH

if [[ -d .venv ]]; then
  echo "[INFO] Removing existing .venv"
  rm -rf .venv
fi

echo "[INFO] Creating fresh .venv"
python3 -m venv .venv

source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .

echo "[INFO] Virtual environment configured: ${ROOT_DIR}/.venv"
