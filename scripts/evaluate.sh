#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/lib/launch.sh
TEST_DATA="${TEST_DATA:-data/splits/test.jsonl}"
RAG_MODE="${RAG_MODE:-always}"
RAG_INDEX="${RAG_INDEX:-artifacts/retrieval}"
RAG_TOKENIZER="${RAG_TOKENIZER:-${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}}"
OUT_DIR="${OUT_DIR:-artifacts/evaluation/run_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
rag_preflight=(); rag_runtime=(--rag-mode "$RAG_MODE")
if [[ "$RAG_MODE" == always ]]; then
  rag_preflight=(--rag-index "$RAG_INDEX" --rag-tokenizer "$RAG_TOKENIZER")
  rag_runtime+=(--rag-index-path "$RAG_INDEX" --rag-tokenizer "$RAG_TOKENIZER" --rag-top-k 3 --rag-max-demos 3 \
    --rag-max-added-tokens 2048 --rag-fusion-weights '0,0.4,0.4,0.2')
elif [[ "$RAG_MODE" != off ]]; then
  printf 'RAG_MODE must be always (paper method) or off (ablation).\n' >&2; exit 2
fi
run_command python3 tools/preflight.py eval --input "$TEST_DATA" "${rag_preflight[@]}" --imports
if [[ "$RAG_MODE" == always ]]; then
  run_command python3 tools/preflight.py rag --input "$RAG_INDEX/documents.json" --imports
fi
prepare_output
run_command python3 scripts/evaluate_fixes.py \
  --model "${SERVED_MODEL:-adapter}" --base-url "${OPENAI_BASE_URL:-http://localhost:8000/v1}" \
  --db-path sqlite:///data/context.sqlite3 --jsonl-path "$TEST_DATA" \
  --n-samples 10 --temperature 0.7 --seed 0 --max-tokens 1024 --use-jsonl-prompt \
  --enable-mythril-check --mythril-bin "${MYTHRIL_BIN:-myth}" --mythril-timeout 30 \
  --force-single-n --llm-request-workers 2 --sample-workers 1 --sample-executor process \
  "${rag_runtime[@]}" \
  --output "$OUT_DIR/evaluation.json" "${EXTRA_ARGS[@]}"
