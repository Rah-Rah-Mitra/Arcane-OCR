#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SETUP_DIR="${ROOT_DIR}/setup-files"

# Prevent package discovery from other repos accidentally injected via PYTHONPATH
unset PYTHONPATH

DRIVER_DEB="${SETUP_DIR}/hailort-pcie-driver_4.23.0_all.deb"
HAILORT_DEB="${SETUP_DIR}/hailort_4.23.0_arm64.deb"
TAPPAS_DEB="${SETUP_DIR}/hailo-tappas-core_5.2.0_arm64.deb"
HAILORT_WHL="${SETUP_DIR}/hailort-4.23.0-cp311-cp311-linux_aarch64.whl"
TAPPAS_WHL="${SETUP_DIR}/hailo_tappas_core_python_binding-5.2.0-py3-none-any.whl"

for f in "$DRIVER_DEB" "$HAILORT_DEB" "$TAPPAS_DEB" "$HAILORT_WHL" "$TAPPAS_WHL"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] Missing required file: $f"
    exit 1
  fi
done

echo "[INFO] Installing Hailo runtime .deb packages"
sudo apt install -y "$DRIVER_DEB" "$HAILORT_DEB" "$TAPPAS_DEB"

echo "[INFO] Verifying linker cache"
ldconfig -p | grep -i libhailort || true

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[WARN] No active virtual environment detected."
  echo "[WARN] Activate your project venv first, then rerun to install wheels there:"
  echo "       source .venv/bin/activate"
  exit 0
fi

echo "[INFO] Installing Python bindings in active venv: ${VIRTUAL_ENV}"
python -m pip install --upgrade "$HAILORT_WHL" "$TAPPAS_WHL"

echo "[INFO] Runtime installation complete"
