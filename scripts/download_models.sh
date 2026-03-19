#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${ROOT_DIR}/models/hailo8l"
mkdir -p "$MODEL_DIR"

DET_URL="https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8l/ocr_det.hef"
REC_URL="https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8l/ocr.hef"

download_file() {
  local url="$1"
  local target="$2"
  if [[ -f "$target" ]]; then
    echo "[INFO] Exists: $target"
    return 0
  fi
  echo "[INFO] Downloading $(basename "$target")"
  curl -fL --retry 3 --retry-delay 2 -o "$target" "$url"
}

download_file "$DET_URL" "${MODEL_DIR}/ocr_det.hef"
download_file "$REC_URL" "${MODEL_DIR}/ocr.hef"

echo "[INFO] Models available under ${MODEL_DIR}"
