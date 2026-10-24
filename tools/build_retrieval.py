#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _code_tokens(code):
    parts = re.findall(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/|[A-Za-z_$][\w$]*|\d+|[^\s]',
                       code or "", flags=re.S)
    return [p for p in parts if not p.startswith(("//", "/*"))]


def test_overlap_indices(documents, test_path, threshold=0.8):
    import sqlite3
    if not test_path.is_file():
        raise FileNotFoundError(f"Test set required for isolation check: {test_path}")
    samples = [json.loads(line) for line in test_path.read_text().splitlines() if line.strip()]
    ids = {str(s.get("id", "")) for s in samples} - {""}
    texts = []
    for sample in samples:
        code = sample.get("input", "").partition("**Source Code:**")[2]
        code = code.split("Please provide the complete fixed version", 1)[0]
        if code.strip(): texts.append(code)
        output = sample.get("output", "")
        match = re.search(r"```(?:solidity)?\s*(.*?)```", output, re.S)
        if output.strip(): texts.append(match.group(1) if match else output)
    root = test_path.resolve().parents[2]
    origins = set()
    context = root / "data/context.sqlite3"
    if context.is_file():
        with sqlite3.connect(f"file:{context}?mode=ro", uri=True) as db:
            for ident, code, path in db.execute("SELECT id,function_code,contract_path FROM smart_contract_functions"):
                ids.add(f"fix_{ident}")
                texts.append(code)
                origins.add(path)
            for path in origins:
                source = root / path
                if source.is_file(): texts.append(source.read_text())
        provenance = root / "data/contracts.json"
        if provenance.is_file():
            for contract in json.loads(provenance.read_text()):
                origins.update(contract.get("upstream_paths", []))
    known = []
    seen = set()
    for code in texts:
        tokens = _code_tokens(code)
        sequence = " " + " ".join(tokens) + " "
        if tokens and sequence not in seen:
            seen.add(sequence); known.append((sequence, set(tokens)))
    rejected = []
    for index, doc in enumerate(documents):
        meta = doc.get("metadata") or {}
        if not isinstance(meta, dict): meta = {}
        if any(str(obj.get(k, "")) in ids for obj in (doc, meta) for k in ("id", "sample_id", "source_id")):
            rejected.append(index); continue
        if any(str(obj.get(k, "")) in origins for obj in (doc, meta) for k in ("contract_path", "source_contract", "upstream_path")):
            rejected.append(index); continue
        overlap = False
        for key in ("code", "fixed_code"):
            tokens = _code_tokens(doc.get(key, ""))
            if not tokens: continue
            sequence, token_set = " " + " ".join(tokens) + " ", set(tokens)
            for other, other_set in known:
                if sequence == other or other in sequence or sequence in other:
                    overlap = True; break
                if min(len(token_set), len(other_set)) <= threshold * max(len(token_set), len(other_set)):
                    continue
                union = token_set | other_set
                if union and len(token_set & other_set) / len(union) > threshold:
                    overlap = True; break
            if overlap: break
        if overlap: rejected.append(index)
    return rejected


def main() -> None:
    parser = argparse.ArgumentParser(description='Build a retrieval index from a user-supplied corpus, separate from the test set.')
    parser.add_argument("--documents", type=Path, default=os.environ.get("RETRIEVAL_DOCUMENTS") or None,
                        help="Your retrieval JSON corpus; this test-only release does not include demonstrations")
    parser.add_argument("--output", type=Path, default=Path(os.environ.get("RAG_INDEX", "artifacts/retrieval")))
    parser.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Hugging Face identifier or local model path")
    parser.add_argument("--fusion-weights", default="0,0.4,0.4,0.2", help="token overlap,BM25,dense,structural")
    parser.add_argument("--mmr-lambda", type=float, default=0.7)
    args = parser.parse_args()
    if args.documents is None:
        parser.error("Pass --documents or set RETRIEVAL_DOCUMENTS to your retrieval corpus; this release contains only the test set")
    args.documents = args.documents.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        parser.error("Output exists; select a new directory to preserve its corpus and configuration")
    if not args.documents.is_file():
        parser.error("Missing corpus; pass --documents pointing to your retrieval JSON corpus")
    try:
        weights = tuple(float(value) for value in args.fusion_weights.split(","))
    except ValueError:
        parser.error("--fusion-weights must contain four comma-separated numbers")
    if len(weights) != 4 or any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        parser.error("--fusion-weights must contain four finite nonnegative values with a positive sum")
    if not 0 <= args.mmr_lambda <= 1:
        parser.error("--mmr-lambda must be between 0 and 1")
    raw = args.documents.read_bytes()
    try:
        docs = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        parser.error("Retrieval corpus must be a JSON array, not a test/training JSONL file")
    if not isinstance(docs, list) or not docs:
        parser.error("Retrieval corpus must be a nonempty JSON array")
    for index, doc in enumerate(docs):
        if not isinstance(doc, dict) or any(not isinstance(doc.get(key), str) or not doc[key].strip()
                                            for key in ("code", "fixed_code")):
            parser.error(f"Document {index} requires nonempty code and fixed_code strings")
    overlaps = test_overlap_indices(docs, ROOT / "data/splits/test.jsonl")
    if overlaps:
        parser.error(f"Retrieval documents overlap the released test set at indices {overlaps[:10]}; use an independent corpus")
    from src.tools.rag_retriever import HybridRetriever
    retriever = HybridRetriever(fusion_weights=weights, mmr_lambda=args.mmr_lambda)
    retriever.build(docs, dense_model=args.dense_model)
    channels = {"token_overlap": bool(retriever.token.documents), "bm25": retriever._bm25 is not None,
                "dense": retriever._dense_index is not None, "structural": retriever._lsh is not None}
    missing = [name for weight, (name, available) in zip(weights, channels.items()) if weight > 0 and not available]
    if missing:
        raise RuntimeError("Missing retrieval channels: " + ", ".join(missing) + "; run tools/preflight.py rag --imports")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".retrieval-", dir=args.output.parent))
    try:
        retriever.save(str(staging))
        staging.rename(args.output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"Saved retrieval index with {len(docs)} documents to {args.output}")


if __name__ == "__main__":
    main()
