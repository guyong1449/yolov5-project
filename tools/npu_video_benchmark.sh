#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS="${WEIGHTS:-${ROOT_DIR}/checkpoint/yolov5_best.pt}"
DATA_YAML="${DATA_YAML:-${ROOT_DIR}/data/dataAirVis.yaml}"
SOURCE_PATH="${SOURCE_PATH:-/root/workspace/data/videos}"
PROJECT_DIR="${PROJECT_DIR:-/root/workspace/outputs/runs}"
RUN_NAME="${RUN_NAME:-npu_bench}"
DEVICE="${DEVICE:-npu:0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${PROJECT_DIR}"

exec "${PYTHON_BIN}" "${ROOT_DIR}/detect.py" \
  --weights "${WEIGHTS}" \
  --source "${SOURCE_PATH}" \
  --data "${DATA_YAML}" \
  --device "${DEVICE}" \
  --project "${PROJECT_DIR}" \
  --name "${RUN_NAME}"
