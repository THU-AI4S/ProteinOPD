#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_MODEL_DIR="$(cd "${REPO_ROOT}/.." && pwd)"
export CUDA_VISIBLE_DEVICES=6,7
export WANDB_PROJECT="prollama_lora_teacher"
#可训练参数数量：19,988,480 rank=8，alpha=16
NPROC_PER_NODE=2
MASTER_PORT=29508

PRETRAINED_MODEL="/path/to/ProLLaMA_model"
TRAIN_FILE="/path/to/train_instruction.json"
VALIDATION_FILE="/path/to/validation_instruction.json"
OUTPUT_DIR="${REPO_ROOT}/trained_adapter/lora/prollama_teacher_lr2e-5_rank8"

LR=2e-5
LORA_RANK=8
LORA_ALPHA=16
LORA_TRAINABLE="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
#LORA_TRAINABLE="q_proj,v_proj"
LORA_DROPOUT=0.1

PER_DEVICE_TRAIN_BATCH_SIZE=8
PER_DEVICE_EVAL_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=2
MAX_SEQ_LENGTH=512
NUM_TRAIN_EPOCHS=8
LOGGING_STEPS=5
EVAL_STEPS=200
SAVE_STEPS=10
SAVE_TOTAL_LIMIT=25
DEEPSPEED_CONFIG_FILE="${SCRIPT_DIR}/ds_zero2_no_offload.json"

cd "${SCRIPT_DIR}"

if python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 1)"; then
  PRECISION_FLAG="--bf16"
  TORCH_DTYPE="bfloat16"
else
  PRECISION_FLAG="--fp16"
  TORCH_DTYPE="float16"
fi

torchrun --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" instruction_tune.py \
  --deepspeed "${DEEPSPEED_CONFIG_FILE}" \
  --model_name_or_path "${PRETRAINED_MODEL}" \
  --tokenizer_name_or_path "${PRETRAINED_MODEL}" \
  --train_file "${TRAIN_FILE}" \
  --validation_file "${VALIDATION_FILE}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --do_train \
  --do_eval \
  --seed 42 \
  ${PRECISION_FLAG} \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --lr_scheduler_type cosine \
  --learning_rate "${LR}" \
  --warmup_ratio 0.05 \
  --weight_decay 0 \
  --logging_strategy steps \
  --logging_steps "${LOGGING_STEPS}" \
  --eval_strategy steps \
  --eval_steps "${EVAL_STEPS}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --preprocessing_num_workers 8 \
  --max_seq_length "${MAX_SEQ_LENGTH}" \
  --output_dir "${OUTPUT_DIR}" \
  --ddp_timeout 30000 \
  --logging_first_step True \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --trainable "${LORA_TRAINABLE}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --torch_dtype "${TORCH_DTYPE}" \
  --load_in_kbits 16 \
  --save_safetensors False \
  --ddp_find_unused_parameters False \
  --gradient_checkpointing
