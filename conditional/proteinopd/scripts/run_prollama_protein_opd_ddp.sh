#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export CUDA_VISIBLE_DEVICES=6,7
export WANDB_PROJECT="prollama_geometric_opd"

NPROC_PER_NODE=2
MASTER_PORT=29519

OPD_CONFIG_PATH="${REPO_ROOT}/conditional/proteinopd/configs/prollama_protein_opd_example.yaml"
OUTPUT_DIR="${REPO_ROOT}/outputs/prollama_geometric_opd"
DEEPSPEED_CONFIG_FILE="${REPO_ROOT}/conditional/teacher_construct/scripts/ds_zero2_no_offload.json"

PER_DEVICE_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=2
LEARNING_RATE=2e-5
LOGGING_STEPS=5
SAVE_STEPS=5
SAVE_TOTAL_LIMIT=20

cd "${REPO_ROOT}"

if python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 1)"; then
  PRECISION_FLAG="--bf16"
else
  PRECISION_FLAG="--fp16"
fi

torchrun --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" \
  "${REPO_ROOT}/conditional/proteinopd/prollama_opd_train.py" \
  --deepspeed "${DEEPSPEED_CONFIG_FILE}" \
  --opd_config_path "${OPD_CONFIG_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --do_train \
  --seed 42 \
  ${PRECISION_FLAG} \
  --num_train_epochs 1.0 \
  --lr_scheduler_type cosine \
  --learning_rate "${LEARNING_RATE}" \
  --warmup_ratio 0.1 \
  --weight_decay 0 \
  --logging_strategy steps \
  --logging_steps "${LOGGING_STEPS}" \
  --report_to wandb \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --eval_strategy no \
  --load_in_kbits 16 \
  --save_safetensors False \
  --ddp_find_unused_parameters False \
  --gradient_checkpointing
