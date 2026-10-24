#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    "eval": {"sqlalchemy": "SQLAlchemy", "openai": "openai", "nltk": "nltk",
             "tqdm": "tqdm", "slither": "slither-analyzer", "solcx": "py-solc-x"},
    "rag": {"rank_bm25": "rank-bm25", "sentence_transformers": "sentence-transformers",
            "numpy": "numpy", "faiss": "faiss-cpu", "datasketch": "datasketch", "tiktoken": "tiktoken"},
    "sft": {"torch": "torch", "transformers": "transformers", "peft": "peft",
            "accelerate": "accelerate", "bitsandbytes": "bitsandbytes", "numpy": "numpy",
            "wandb": "wandb", "tqdm": "tqdm"},
}
MODULES["rl"] = {**MODULES["eval"], **MODULES["rag"], **MODULES["sft"]}


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def validate_tokenizer(folder: Path) -> str | None:
    if not folder.is_dir() or not (folder / "tokenizer_config.json").is_file():
        return "RAG_TOKENIZER must name a local model/tokenizer directory containing tokenizer_config.json"
    if not any((folder / filename).is_file() for filename in (
        "tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json", "vocab.txt"
    )):
        return "RAG_TOKENIZER is missing its vocabulary/tokenizer assets"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description='Check local inputs and dependencies without model downloads or API calls.')
    parser.add_argument("mode", choices=tuple(MODULES))
    parser.add_argument("--input", action="append", default=[], help="Input file to check; replaces mode-specific data defaults")
    parser.add_argument("--adapter", type=Path, help="Local LoRA checkpoint to validate")
    parser.add_argument("--rag-index", type=Path, help="Require an existing retrieval index")
    parser.add_argument("--rag-tokenizer", type=Path, help="Local tokenizer used to enforce the retrieval token budget")
    parser.add_argument("--context-db", type=Path, help="Training-context SQLite file; required for RL")
    parser.add_argument("--mythril-bin", default=os.environ.get("MYTHRIL_BIN", "myth"))
    parser.add_argument("--imports", action="store_true", help="Also import packages to detect binary/transitive dependency failures")
    parser.add_argument("--cuda-devices", type=int, help="Require this many visible CUDA GPUs (imports torch; no model loading)")
    args = parser.parse_args()
    errors = []
    missing_modules = []
    versions = {}
    if sys.version_info < (3, 10):
        errors.append("Python 3.10 or newer is required")
    for module, package in MODULES[args.mode].items():
        if importlib.util.find_spec(module) is None:
            missing_modules.append(module)
        else:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "unavailable"
    if missing_modules:
        errors.append("Install the dependencies listed in README.md" +
                      (", including training dependencies" if args.mode in ("sft", "rl") else ""))
    commands = ["accelerate"] if args.mode in ("sft", "rl") else []
    if args.mode in ("eval", "rl"):
        commands.append(args.mythril_bin)
    missing_commands = [command for command in commands if shutil.which(command) is None]
    if missing_commands:
        errors.append("Missing executable(s): " + ", ".join(missing_commands) +
                      ". Install accelerate for training; install Mythril separately and export MYTHRIL_BIN")

    default_inputs = {
        "sft": [os.environ.get("TRAIN_DATA"), os.environ.get("VAL_DATA")],
        "rl": [os.environ.get("TRAIN_DATA")],
        "eval": [os.environ.get("TEST_DATA", "data/splits/test.jsonl")],
        "rag": [os.environ.get("RETRIEVAL_DOCUMENTS")],
    }
    input_values = [value for value in (args.input or default_inputs[args.mode]) if value]
    if not input_values:
        errors.append("This release contains only test data. Supply your own training/retrieval corpus with --input or the documented environment variable")
    inputs = [project_path(value) for value in input_values]
    if args.mode == "eval":
        inputs.append(ROOT / "data/context.sqlite3")
    elif args.mode == "rl":
        context_db = args.context_db or os.environ.get("CONTEXT_DB")
        if context_db:
            inputs.append(project_path(context_db))
        else:
            errors.append("RL requires --context-db or CONTEXT_DB for your training samples; the released context database covers test samples only")
    missing_inputs = [str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
                      for path in inputs if not path.is_file() or path.stat().st_size == 0]
    if missing_inputs:
        errors.append("Missing/empty input(s). This release contains only test data; provide your own training/retrieval inputs")
    if args.adapter is not None:
        adapter = project_path(args.adapter)
        if not (adapter / "adapter_config.json").is_file() or not any(
            (adapter / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
        ):
            errors.append("--adapter must contain adapter_config.json and adapter_model.safetensors or adapter_model.bin")
    if args.rag_index is not None:
        index = project_path(args.rag_index)
        if not (index / "index_metadata.json").is_file() or not (index / "documents.json").is_file():
            errors.append("Retrieval index missing; run: python3 tools/build_retrieval.py --output " + str(args.rag_index))
    if args.rag_tokenizer is not None:
        tokenizer_error = validate_tokenizer(project_path(args.rag_tokenizer))
        if tokenizer_error:
            errors.append(tokenizer_error)

    import_failures = []
    if args.imports:
        for module in MODULES[args.mode]:
            if module in missing_modules:
                continue
            try:
                result = subprocess.run([sys.executable, "-c", f"import {module}"],
                                        capture_output=True, text=True, timeout=30)
                if result.returncode:
                    import_failures.append({"module": module, "error": result.stderr[-1600:]})
            except subprocess.TimeoutExpired:
                import_failures.append({"module": module, "error": "import timed out"})
        if import_failures:
            errors.append("Package import failed; repair the environment before running")
    cuda = None
    if args.cuda_devices is not None:
        if args.cuda_devices < 1:
            parser.error("--cuda-devices must be positive")
        try:
            import torch
            cuda = {"available": torch.cuda.is_available(), "visible_devices": torch.cuda.device_count()}
            if not cuda["available"] or cuda["visible_devices"] < args.cuda_devices:
                errors.append(f"Need {args.cuda_devices} visible CUDA devices; check CUDA_VISIBLE_DEVICES and the PyTorch build")
        except Exception as exc:
            errors.append(f"Cannot initialize PyTorch/CUDA: {type(exc).__name__}: {exc}")
    print(json.dumps({"mode": args.mode,
                      "status": "FAIL" if errors else "PASS", "versions": versions,
                      "missing_modules": missing_modules, "missing_commands": missing_commands,
                      "missing_inputs": missing_inputs, "import_failures": import_failures,
                      "cuda": cuda, "errors": errors}, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
