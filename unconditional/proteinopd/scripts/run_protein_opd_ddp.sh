#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

accelerate launch \
  --config_file "${REPO_ROOT}/unconditional/proteinopd/accelerate_ddp.yaml" \
  --num_processes 2 \
  "${REPO_ROOT}/unconditional/proteinopd/protein_opd_train.py" \
  --num_train_samples 8192 \
  --student_model_name_or_path /path/to/protgpt2-student \
  --teacher_config_path "${REPO_ROOT}/unconditional/proteinopd/configs/teachers.yaml" \
  --prompt_mode unconditional \
  --max_new_tokens 512 \
  --temperature 1.0 \
  --repetition_penalty 1.2 \
  --top_p 0.95 \
  --top_k 500 \
  --beta 0.5 \
  --top_k_loss 0 \
  --useloss jsd \
  --use_entropy_rule \
  --entropy_alpha 0.0 \
  --student_tune_mode lora \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --lora_target_modules c_attn c_proj c_fc \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-5 \
  --num_train_epochs 1.0 \
  --logging_steps 1 \
  --save_steps 100 \
  --report_to wandb \
  --run_name protein_opd_jsd_lr2e-5 \
  --output_dir "${REPO_ROOT}/outputs/protein_opd_jsd_lr2e-5" \
  --warmup_ratio 0.1
