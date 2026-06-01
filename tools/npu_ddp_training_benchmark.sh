#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Load .env if present
set -a; [ -f "${ROOT_DIR}/.env" ] && source "${ROOT_DIR}/.env"; set +a

DATA_YAML="${DATA_YAML:-${YOLO_DDP_DATA_YAML:-${ROOT_DIR}/data/dataAirVis.yaml}}"
WEIGHTS="${WEIGHTS:-${ROOT_DIR}/checkpoint/yolov5_best.pt}"
PROJECT_DIR="${PROJECT_DIR:-runs/train}"
RUN_NAME="${RUN_NAME:-panel_smoke10_ddp}"
DEVICE="${DEVICE:-${YOLO_DEVICE:-npu:0,1,2,3}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-${YOLO_MASTER_PORT:-29501}}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-0}"
SEED="${SEED:-0}"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export ASCEND_RT_VISIBLE_DEVICES

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_port "${MASTER_PORT}" \
  "${ROOT_DIR}/train.py" \
  --data "${DATA_YAML}" \
  --weights "${WEIGHTS}" \
  --imgsz "${IMGSZ}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --workers "${WORKERS}" \
  --project "${PROJECT_DIR}" \
  --name "${RUN_NAME}" \
  --seed "${SEED}"
