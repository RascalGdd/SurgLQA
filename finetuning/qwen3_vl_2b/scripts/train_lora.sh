#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

: "${SURGLQA_ANNOTATION:?Set SURGLQA_ANNOTATION to the training JSON or JSONL file}"
: "${SURGLQA_MEDIA_ROOT:?Set SURGLQA_MEDIA_ROOT to the image/video root directory}"

if [[ ! -f "${SURGLQA_ANNOTATION}" ]]; then
  echo "Annotation file not found: ${SURGLQA_ANNOTATION}" >&2
  exit 1
fi

if [[ ! -d "${SURGLQA_MEDIA_ROOT}" ]]; then
  echo "Media directory not found: ${SURGLQA_MEDIA_ROOT}" >&2
  exit 1
fi

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/qwen3_vl_2b_lora}"
CACHE_DIR="${CACHE_DIR:-${PROJECT_DIR}/.cache}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l | tr -d ' ')}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  qwenvl/train/train_qwen.py \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_use "surglqa_train%100" \
  --output_dir "${OUTPUT_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --bf16 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing True \
  --learning_rate 1e-4 \
  --optim adamw_torch \
  --model_max_length 4096 \
  --data_flatten True \
  --data_packing False \
  --max_pixels 50176 \
  --min_pixels 784 \
  --video_fps 2 \
  --video_max_frames 8 \
  --video_min_frames 4 \
  --video_max_pixels 1304576 \
  --video_min_pixels 200704 \
  --num_train_epochs 3 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --weight_decay 0.01 \
  --logging_steps 10 \
  --save_steps 1000 \
  --save_total_limit 3 \
  --lora_enable True \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.0 \
  --deepspeed scripts/zero3.json \
  "$@"
