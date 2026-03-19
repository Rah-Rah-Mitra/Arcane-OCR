#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Keep runtime isolated from previously sourced external environments
unset PYTHONPATH

# Enable Hailo monitor telemetry by default for runtime verification.
# Users can override by exporting HAILO_MONITOR=0 before running this script.
export HAILO_MONITOR="${HAILO_MONITOR:-1}"

if [[ ! -d .venv ]]; then
  echo "[ERROR] .venv not found. Run scripts/setup_venv.sh first."
  exit 1
fi

source .venv/bin/activate
python -m arcane_ocr.pipeline "$@"
