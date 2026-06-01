#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "${ROOT_DIR}/../.." && pwd)"
# Load .env if present
set -a; [ -f "${ROOT_DIR}/.env" ] && source "${ROOT_DIR}/.env"; set +a

WEIGHTS="${WEIGHTS:-${ROOT_DIR}/checkpoint/yolov5_best.pt}"
DATA_YAML="${DATA_YAML:-${ROOT_DIR}/data/dataAirVis.yaml}"
SOURCE_PATH="${SOURCE_PATH:-${YOLO_VIDEO_DIR:-${WORKSPACE_DIR}/data/videos}/video20.mp4}"
PROJECT_DIR="${PROJECT_DIR:-${YOLO_OUTPUT_ROOT:-${WORKSPACE_DIR}/outputs/runs}}"
RUN_NAME="${RUN_NAME:-npu_video20_ddp}"
DEVICE="${DEVICE:-${YOLO_DEVICE:-npu:0,1,2,3}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29541}"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${PROJECT_DIR}"
export ASCEND_RT_VISIBLE_DEVICES
export NPROC_PER_NODE

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_port "${MASTER_PORT}" \
  "${ROOT_DIR}/detect.py" \
  --weights "${WEIGHTS}" \
  --source "${SOURCE_PATH}" \
  --data "${DATA_YAML}" \
  --device "${DEVICE}" \
  --project "${PROJECT_DIR}" \
  --name "${RUN_NAME}" \
  --ddp-infer \
  --frame-shard-mode mod \
  --save-summary-only \
  --nosave
