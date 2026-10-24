#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/lib/launch.sh
: "${SFT_ADAPTER:?Set SFT_ADAPTER to the final SFT adapter directory}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
validate_processes
if [[ "$NUM_PROCESSES" != 2 && "$NUM_PROCESSES" != 4 ]]; then
  printf 'Paired-reference RL requires NUM_PROCESSES=2 or 4.\n' >&2; exit 2
fi
: "${TRAIN_DATA:?Set TRAIN_DATA to your SFT training JSONL; this release contains the test set only}"
: "${CONTEXT_DB:?Set CONTEXT_DB to your training-context SQLite file; the released database covers test samples only}"
DB_PATH="${DB_PATH:-sqlite:///$CONTEXT_DB}"
if [[ "$DB_PATH" != "sqlite:///$CONTEXT_DB" ]]; then
  printf 'DB_PATH must equal sqlite:///$CONTEXT_DB. Set CONTEXT_DB to the desired SQLite file.\n' >&2
  exit 2
fi
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
OUT_DIR="${OUT_DIR:-artifacts/rl/run_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
run_command python3 tools/preflight.py rl --input "$TRAIN_DATA" --context-db "$CONTEXT_DB" --adapter "$SFT_ADAPTER" --imports --cuda-devices "$NUM_PROCESSES"
prepare_output
run_command accelerate launch --multi_gpu --num_processes "$NUM_PROCESSES" --mixed_precision bf16 \
  scripts/rl/train_lora_grpo_pairedref.py --ref-mode paired \
  --train-jsonl "$TRAIN_DATA" --limit 0 --base-model "$BASE_MODEL" \
  --sft-lora "$SFT_ADAPTER" --reference-lora "$SFT_ADAPTER" \
  --k 4 --steps 400 --batch-prompts 1 --lr 2e-5 --out-dir "$OUT_DIR" \
  --prompt-format chat --use-db-contract --db-path "$DB_PATH" \
  --dtype bf16 --gradient-checkpointing --epsilon-cost 1.25 --lambda-lr 0.01 \
  --grad-accum 2 --lambda-init 0 --gen-microbatch 4 --seed 1 --mythril \
  --mythril-bin "${MYTHRIL_BIN:-myth}" --mythril-timeout 30 --rag-mode off --max-new-tokens 1024 "${EXTRA_ARGS[@]}"
