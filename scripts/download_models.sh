#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${ROOT_DIR}/models/hailo8l"
SAMPLES_DIR="${ROOT_DIR}/public/samples"
mkdir -p "$MODEL_DIR"
mkdir -p "$SAMPLES_DIR"

DET_URL="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/paddle_ocr_v5_mobile_detection.hef"
REC_URL="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/paddle_ocr_v5_mobile_recognition.hef"
DICT_URL="https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/dict/ppocrv5_dict.txt"

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
download_file "$DICT_URL" "${SAMPLES_DIR}/ppocrv5_dict.txt"

echo "[INFO] Models available under ${MODEL_DIR}"
