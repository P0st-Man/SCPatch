# SCPatch

Code and the 175-case test set for *SCPatch: Automated Smart Contract Vulnerability Repair via Security-Constrained Policy Optimization*.

## Data

- `data/splits/test.jsonl`: 175 tasks with prompts and reference repairs. Output the target function, or the complete contract when `repair_scope` is `contract`.
- `data/context.sqlite3`: test-function context, indexed by the integer in `fix_<id>`.
- `data/contracts/`: 154 source contracts for verification; sources are listed in `data/contracts.json`.

## Installation

Use Python 3.10–3.12:

```bash
python3 -m pip install \
  'SQLAlchemy>=2.0,<3' 'openai>=1.0,<3' 'nltk>=3.8,<4' 'tqdm>=4.65,<5' \
  'py-solc-x>=2.0,<3' 'slither-analyzer==0.10.3' 'semantic-version>=2.10,<3' \
  'numpy>=1.26,<2' 'sentence-transformers>=3,<6' 'transformers>=4.46,<5' \
  'rank-bm25>=0.2.2,<0.3' 'datasketch>=1.6,<2' 'faiss-cpu>=1.8,<2' 'tiktoken>=0.7,<1'
```

Install Mythril in a separate Python 3.11 environment:

```bash
python3.11 -m venv .venv-mythril
.venv-mythril/bin/python -m pip install 'mythril==0.24.8'
export MYTHRIL_BIN="$PWD/.venv-mythril/bin/myth"
```

## Evaluation

Set the model service address and served model name:

```bash
export MODEL_BASE_URL=http://localhost:8000/v1
export SERVED_MODEL=scpatch
```

Retrieval examples use this JSON format: `[{"code": "...", "fixed_code": "..."}]`.

```bash
python3 tools/build_retrieval.py \
  --documents /path/to/examples.json --output artifacts/retrieval
RAG_INDEX=artifacts/retrieval RAG_TOKENIZER=/path/to/model/tokenizer \
  bash scripts/evaluate.sh
```

## Training

Install the additional dependencies:

```bash
python3 -m pip install 'torch>=2.6,<3' 'peft>=0.13,<1' 'accelerate>=1,<2' \
  'bitsandbytes>=0.44,<1' 'wandb>=0.16,<1' 'datasets>=2.14,<5' 'tabulate>=0.9,<1'
```

```bash
TRAIN_DATA=/path/to/train.jsonl NUM_PROCESSES=4 bash scripts/run_sft.sh

TRAIN_DATA=/path/to/train.jsonl CONTEXT_DB=/path/to/training-context.sqlite3 \
  SFT_ADAPTER=/path/to/sft-adapter NUM_PROCESSES=4 bash scripts/run_rl.sh
```

