#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/lib/launch.sh
export WANDB_MODE="${WANDB_MODE:-disabled}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
validate_processes
: "${TRAIN_DATA:?Set TRAIN_DATA to your training JSONL; this release contains the test set only}"
VAL_DATA="${VAL_DATA:-}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
OUT_DIR="${OUT_DIR:-artifacts/sft/run_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
val_input=(); val_train=()
if [[ -n "$VAL_DATA" ]]; then
  val_input=(--input "$VAL_DATA"); val_train=(--val_data "$VAL_DATA")
fi
run_command python3 tools/preflight.py sft --input "$TRAIN_DATA" "${val_input[@]}" --imports --cuda-devices "$NUM_PROCESSES"
prepare_output
launch=(accelerate launch --num_processes "$NUM_PROCESSES" --mixed_precision fp16)
if (( NUM_PROCESSES > 1 )); then launch+=(--multi_gpu); fi
run_command "${launch[@]}" scripts/sft/train_lora.py \
  --train_data "$TRAIN_DATA" "${val_train[@]}" --model_path "$BASE_MODEL" \
  --output_dir "$OUT_DIR" --batch_size 1 --gradient_accumulation 8 \
  --learning_rate 5e-5 --warmup_steps 2 --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --epochs 2 --fp16 --device cuda --quantization 8bit --prompt_format chat "${EXTRA_ARGS[@]}"
