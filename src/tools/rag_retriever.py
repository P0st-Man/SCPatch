from __future__ import annotations

import hashlib
import json
import logging
import math
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    code: str
    fixed_code: str
    vulnerability_type: str
    score: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z_]\w+|\d+|[^\s]", text or "")]


class TokenOverlapIndex:

    def __init__(self) -> None:
        self.documents: List[Dict[str, Any]] = []
        self.doc_sets: List[set] = []

    def build(self, documents: List[Dict[str, Any]]) -> None:
        self.documents = list(documents)
        self.doc_sets = [set(_tokenize(d.get("code", ""))) for d in self.documents]
        logger.info("[rag] token_overlap built docs=%d", len(self.documents))

    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        qset = set(_tokenize(query))
        if not qset:
            return []
        qn = float(len(qset)) ** 0.5
        scored: List[Tuple[float, int]] = []
        for i, dset in enumerate(self.doc_sets):
            inter = len(qset & dset)
            if inter <= 0:
                continue
            dn = float(len(dset)) ** 0.5 if dset else 1.0
            s = float(inter) / max(1e-9, (qn * dn))
            scored.append((s, i))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[RetrievalResult] = []
        for s, i in scored[:top_k]:
            d = self.documents[i]
            out.append(
                RetrievalResult(
                    code=d.get("code", ""),
                    fixed_code=d.get("fixed_code", ""),
                    vulnerability_type=str(d.get("vulnerability_type", "") or ""),
                    score=float(s),
                    source="token_overlap",
                    metadata=dict(d),
                )
            )
        return out

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"documents": self.documents}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.build(list(obj.get("documents", [])))


def _mmr_select(
    results: List[RetrievalResult],
    k: int,
    mmr_lambda: float = 0.7,
) -> List[RetrievalResult]:
    if not results:
        return []
    if len(results) <= k:
        return results[:k]
    from difflib import SequenceMatcher

    selected = [results[0]]
    remaining = results[1:]
    while len(selected) < k and remaining:
        best_score = float("-inf")
        best_idx = 0
        for i, r in enumerate(remaining):
            max_sim = max(SequenceMatcher(None, r.code, s.code).ratio() for s in selected)
            mmr = float(mmr_lambda) * float(r.score) - (1.0 - float(mmr_lambda)) * float(max_sim)
            if mmr > best_score:
                best_score = mmr
                best_idx = i
        selected.append(remaining.pop(best_idx))
    return selected


class HybridRetriever:

    def __init__(
        self,
        fusion_weights: Tuple[float, float, float, float] = (0.0, 0.4, 0.4, 0.2),
        mmr_lambda: float = 0.7,
    ) -> None:
        if len(fusion_weights) != 4 or any(not math.isfinite(w) or w < 0 for w in fusion_weights) or sum(fusion_weights) <= 0:
            raise ValueError("Expected four finite nonnegative retrieval weights with a positive sum")
        if not 0 <= float(mmr_lambda) <= 1:
            raise ValueError("mmr_lambda must be between 0 and 1")
        self.weights = tuple(float(w) for w in fusion_weights)
        self.mmr_lambda = float(mmr_lambda)
        self.dense_model = "sentence-transformers/all-MiniLM-L6-v2"

        self.token = TokenOverlapIndex()
        self._bm25 = None
        self._dense = None
        self._struct = None
        self._docs: List[Dict[str, Any]] = []

        try:
            from rank_bm25 import BM25Okapi
        except Exception:
            BM25Okapi = None
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            SentenceTransformer = None
        try:
            import numpy as np
        except Exception:
            np = None
        try:
            import faiss
        except Exception:
            faiss = None
        try:
            from datasketch import MinHash, MinHashLSH
        except Exception:
            MinHash = None
            MinHashLSH = None

        self._BM25Okapi = BM25Okapi
        self._SentenceTransformer = SentenceTransformer
        self._np = np
        self._faiss = faiss
        self._MinHash = MinHash
        self._MinHashLSH = MinHashLSH

        self._dense_encoder = None
        self._dense_index = None
        self._dense_dim = None

        self._lsh = None
        self._minhashes: Dict[str, Any] = {}
        self._struct_docs: Dict[str, Dict[str, Any]] = {}

    def build(self, documents: List[Dict[str, Any]], *, dense_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        if not documents:
            raise ValueError("Retrieval corpus must contain at least one document")
        self.dense_model = dense_model
        self._docs = list(documents)
        self.token.build(documents)
        logger.info("[rag] hybrid build start docs=%d", len(self._docs))

        if self.weights[1] > 0 and self._BM25Okapi is not None:
            corpus = [_tokenize(d.get("code", "")) for d in documents]
            self._bm25 = self._BM25Okapi(corpus)
            logger.info("[rag] bm25 enabled")
        else:
            self._bm25 = None
            logger.info("[rag] bm25 disabled (missing dep rank_bm25)")

        if self.weights[2] > 0 and self._SentenceTransformer is not None and self._faiss is not None and self._np is not None:
            try:
                self._dense_encoder = self._SentenceTransformer(dense_model)
                codes = [d.get("code", "") for d in documents]
                emb = self._dense_encoder.encode(codes, show_progress_bar=False, normalize_embeddings=True)
                emb = self._np.asarray(emb, dtype=self._np.float32)
                self._dense_dim = int(emb.shape[1])
                self._dense_index = self._faiss.IndexHNSWFlat(self._dense_dim, 32)
                self._dense_index.hnsw.efConstruction = 200
                self._dense_index.add(emb)
                self._dense_index.hnsw.efSearch = 50
                logger.info("[rag] dense enabled model=%s dim=%s", dense_model, self._dense_dim)
            except Exception:
                self._dense_encoder = None
                self._dense_index = None
                self._dense_dim = None
                logger.info("[rag] dense disabled (init failed)")
        else:
            self._dense_encoder = None
            self._dense_index = None
            self._dense_dim = None
            logger.info("[rag] dense disabled (missing deps sentence_transformers/faiss/numpy)")

        if self.weights[3] > 0 and self._MinHash is not None and self._MinHashLSH is not None:
            try:
                self._lsh = self._MinHashLSH(threshold=0.5, num_perm=128)
                self._minhashes = {}
                self._struct_docs = {}
                for i, doc in enumerate(documents):
                    doc_id = f"doc_{i}"
                    sig = self._extract_signature(doc.get("code", ""))
                    mh = self._MinHash(num_perm=128)
                    for s in sig:
                        mh.update(s.encode("utf-8"))
                    self._minhashes[doc_id] = mh
                    self._struct_docs[doc_id] = doc
                    self._lsh.insert(doc_id, mh)
                self._struct = True
                logger.info("[rag] structural enabled (minhash)")
            except Exception:
                self._lsh = None
                self._minhashes = {}
                self._struct_docs = {}
                self._struct = None
                logger.info("[rag] structural disabled (init failed)")
        else:
            self._lsh = None
            self._minhashes = {}
            self._struct_docs = {}
            self._struct = None
            logger.info("[rag] structural disabled (missing dep datasketch)")

        self._check_active_channels()

    @property
    def active_channels(self) -> List[str]:
        return [name for name, weight in zip(
            ("token_overlap", "bm25", "dense", "structural"), self.weights
        ) if weight > 0]

    def _check_active_channels(self) -> None:
        available = (bool(self._docs), self._bm25 is not None,
                     self._dense_encoder is not None and self._dense_index is not None,
                     self._lsh is not None)
        missing = [name for name, weight, ready in zip(
            ("token_overlap", "bm25", "dense", "structural"), self.weights, available
        ) if weight > 0 and not ready]
        if missing:
            raise RuntimeError("Requested retrieval channels unavailable: " + ", ".join(missing))

    def search(
        self, query: str, *, top_k: int = 5,
        vuln_type: Optional[str] = None, use_mmr: bool = True,
    ) -> List[RetrievalResult]:
        if int(top_k) <= 0:
            return []
        self._check_active_channels()
        k = min(len(self._docs), int(top_k) * 5)
        candidates: Dict[str, Dict[str, Any]] = {}
        channels: Dict[str, Dict[str, float]] = {}

        def add(name: str, doc: Dict[str, Any], score: float) -> None:
            key = str(doc.get("code", ""))
            candidates.setdefault(key, doc)
            channels.setdefault(name, {})[key] = float(score)

        if self.weights[0] > 0:
            for r in self.token.search(query, top_k=k):
                add("token_overlap", r.metadata, r.score)
        if self.weights[1] > 0:
            scores = self._bm25.get_scores(_tokenize(query))
            for i in sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:k]:
                add("bm25", self._docs[i], float(scores[i]))
        if self.weights[2] > 0:
            q = self._dense_encoder.encode([query], normalize_embeddings=True)
            q = self._np.asarray(q, dtype=self._np.float32)
            distances, indices = self._dense_index.search(q, k)
            for distance, i in zip(distances[0], indices[0]):
                if int(i) >= 0:
                    add("dense", self._docs[int(i)], 1.0 - float(distance) / 2.0)
        if self.weights[3] > 0:
            signature = self._extract_signature(query)
            if signature:
                qmh = self._MinHash(num_perm=128)
                for token in signature:
                    qmh.update(token.encode("utf-8"))
                for doc_id in sorted(self._lsh.query(qmh)):
                    add("structural", self._struct_docs[doc_id], qmh.jaccard(self._minhashes[doc_id]))

        normalized: Dict[str, Dict[str, float]] = {}
        for name, scores in channels.items():
            lo, hi = min(scores.values()), max(scores.values())
            normalized[name] = {key: (value - lo) / (hi - lo) if hi > lo else 1.0
                                for key, value in scores.items()}
        weights = dict(zip(("token_overlap", "bm25", "dense", "structural"), self.weights))
        results = []
        vt = str(vuln_type or "").strip().lower()
        for key, doc in candidates.items():
            if vt and vt != "unknown" and str(doc.get("vulnerability_type", "")).strip().lower() != vt:
                continue
            scores = {name: values.get(key, 0.0) for name, values in normalized.items()}
            metadata = dict(doc)
            metadata["channel_scores"] = scores
            results.append(RetrievalResult(
                code=str(doc.get("code", "")), fixed_code=str(doc.get("fixed_code", "")),
                vulnerability_type=str(doc.get("vulnerability_type", "") or ""),
                score=sum(weights[name] * value for name, value in scores.items()),
                source="hybrid", metadata=metadata,
            ))
        results.sort(key=lambda result: (-result.score, result.code))
        if use_mmr:
            return _mmr_select(results, k=int(top_k), mmr_lambda=self.mmr_lambda)
        return results[:int(top_k)]

    def save(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "documents.json").write_text(json.dumps(self._docs, ensure_ascii=False), encoding="utf-8")
        (p / "index_metadata.json").write_text(json.dumps({
            "schema_version": 1, "dense_model": self.dense_model,
            "weights": list(self.weights), "mmr_lambda": self.mmr_lambda,
            "active_channels": self.active_channels,
        }, indent=2) + "\n", encoding="utf-8")
        logger.info("[rag] saved index to %s docs=%d", str(p), len(self._docs))

    def load(self, path: str) -> None:
        p = Path(path)
        docs = json.loads((p / "documents.json").read_text(encoding="utf-8"))
        metadata_path = p / "index_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        self.build(list(docs), dense_model=metadata.get("dense_model", self.dense_model))
        logger.info("[rag] loaded index from %s docs=%d", str(p), len(self._docs))

    @staticmethod
    def _extract_signature(code: str) -> List[str]:
        patterns = [
            r"function\s+\w+",
            r"modifier\s+\w+",
            r"(if|else|for|while|require)\s*\(",
            r"\.(call|delegatecall|transfer|send)\s*[({]",
            r"emit\s+\w+",
        ]
        sigs: List[str] = []
        for p in patterns:
            for m in re.findall(p, code or ""):
                sigs.append(hashlib.md5(str(m).encode("utf-8", errors="ignore")).hexdigest()[:8])
        return sigs


class RAGPromptBuilder:

    def __init__(self, max_demos: int = 3, max_chars_each: int = 800) -> None:
        self.max_demos = int(max_demos)
        self.max_chars_each = int(max_chars_each)

    def build(self, query_code: str, vuln_info: Dict[str, Any], examples: List[RetrievalResult]) -> str:
        logger.debug("[rag] build prompt demos=%d", len(examples))
        parts: List[str] = []
        for ex in examples[: self.max_demos]:
            parts.append(
                "\n".join(
                    [
                        "<DEMO>",
                        f"### Vulnerable ({ex.vulnerability_type or 'Unknown'}):",
                        "```solidity",
                        (ex.code or "")[: self.max_chars_each],
                        "```",
                        "### Fixed:",
                        "```solidity",
                        (ex.fixed_code or "")[: self.max_chars_each],
                        "```",
                        "</DEMO>",
                    ]
                )
            )

        vuln_types = vuln_info.get("vulnerability_types") or ["Unknown"]
        if not isinstance(vuln_types, list):
            vuln_types = ["Unknown"]
        severity = vuln_info.get("severity", "Unknown")
        parts.append(
            "\n".join(
                [
                    "<QUERY>",
                    f"### Vulnerability: {', '.join([str(v) for v in vuln_types])}",
                    f"### Severity: {severity}",
                    "### Code:",
                    "```solidity",
                    (query_code or "").strip(),
                    "```",
                    "</QUERY>",
                    "",
                    "Provide the fixed code:",
                ]
            )
        )
        return "\n\n".join(parts)


_RE_VULN_TYPES = re.compile(r"vulnerability\s*types?\s*:\s*(.+)", re.IGNORECASE)
_RE_SEVERITY = re.compile(r"severity\s*:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_RE_BULLET_TYPE = re.compile(r"^\s*-\s*type\s*:\s*(.+)\s*$", re.IGNORECASE)
_RE_BULLET_SEVERITY = re.compile(r"^\s*-\s*severity\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.IGNORECASE)
_RE_CODE_FENCE_OUT = re.compile(r"```(?:[a-zA-Z0-9_-]+)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def parse_vuln_info_from_text(text: str) -> Dict[str, Any]:
    vuln_types: List[str] = []
    severity: Any = "Unknown"
    for ln in (text or "").splitlines():
        m0 = _RE_BULLET_TYPE.match(ln)
        if m0:
            raw = m0.group(1)
            parts = [x.strip() for x in re.split(r"[,;/|]", raw) if x.strip()]
            vuln_types.extend(parts)
        m = _RE_VULN_TYPES.search(ln)
        if m:
            raw = m.group(1)
            parts = [x.strip() for x in re.split(r"[,;/|]", raw) if x.strip()]
            vuln_types.extend(parts)
        m00 = _RE_BULLET_SEVERITY.match(ln)
        if m00:
            try:
                severity = float(m00.group(1))
            except Exception:
                severity = m00.group(1)
        m2 = _RE_SEVERITY.search(ln)
        if m2:
            try:
                severity = float(m2.group(1))
            except Exception:
                severity = m2.group(1)
    if not vuln_types:
        vuln_types = ["Unknown"]
    return {"vulnerability_types": vuln_types, "severity": severity}


def extract_solidity_code_block(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    m = _RE_CODE_FENCE_OUT.search(t)
    if m:
        return (m.group(1) or "").strip()
    return t


def extract_function_block(code_or_text: str, function_name: Optional[str] = None) -> str:
    s = extract_solidity_code_block(code_or_text)
    if not s:
        return ""

    def _extract_by_name(name: str) -> str:
        start = s.find(f"function {name}")
        if start == -1:
            return ""
        brace_open = s.find("{", start)
        if brace_open == -1:
            return ""
        i = brace_open
        depth = 0
        while i < len(s):
            ch = s[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1].strip()
            i += 1
        return ""

    if function_name:
        out = _extract_by_name(function_name)
        if out:
            return out

    m = re.search(r"\bfunction\s+([A-Za-z_]\w*)", s)
    if not m:
        return s.strip()
    return _extract_by_name(m.group(1)) or s.strip()


def build_documents_from_fix_sft_jsonl(
    jsonl_path: str,
    *,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:

    def _extract_function_name(inp: str) -> Optional[str]:
        m = re.search(r"\*\*Function:\*\*\s*([A-Za-z_]\w*)", inp or "")
        return m.group(1).strip() if m else None

    def _extract_contract_source(inp: str) -> Optional[str]:
        marker = "**Source Code:**"
        if marker not in (inp or ""):
            return None
        after = (inp or "").split(marker, 1)[1]
        tail = "Please provide the complete fixed version"
        if tail in after:
            after = after.split(tail, 1)[0]
        return after.strip()

    def _extract_original_function(contract: str, function_name: str) -> Optional[str]:
        if not contract or not function_name:
            return None
        start = contract.find(f"function {function_name}")
        if start == -1:
            return None
        brace_open = contract.find("{", start)
        if brace_open == -1:
            return None
        i = brace_open
        depth = 0
        while i < len(contract):
            ch = contract[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return contract[start : i + 1].strip()
            i += 1
        return None

    docs: List[Dict[str, Any]] = []
    n = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue

            inp = str(item.get("input", "") or "")
            out_raw = str(item.get("output", "") or item.get("fixed_code", "") or "")
            out = out_raw
            if not out.strip():
                continue

            func_name = _extract_function_name(inp)
            contract = _extract_contract_source(inp)
            code = ""
            if contract and func_name:
                code = _extract_original_function(contract, func_name) or ""
            if not code:
                code = inp.strip()

            out = extract_function_block(out_raw, function_name=func_name) or extract_solidity_code_block(out_raw)

            vuln_info = parse_vuln_info_from_text(inp)
            vuln_type = ""
            vts = vuln_info.get("vulnerability_types") or []
            if isinstance(vts, list) and vts:
                vuln_type = str(vts[0])

            docs.append(
                {
                    "code": code,
                    "fixed_code": out,
                    "vulnerability_type": vuln_type,
                    "metadata": {
                        "id": item.get("id"),
                        "instruction": item.get("instruction"),
                    },
                }
            )
            n += 1
            if limit and n >= limit:
                break

    logger.info("[rag] built documents from jsonl=%s docs=%d", jsonl_path, len(docs))
    return docs
