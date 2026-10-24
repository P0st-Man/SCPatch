from __future__ import annotations

import difflib
from typing import List


def normalized_edit_distance(a: str, b: str) -> float:
    a, b = a or "", b or ""
    scale = max(len(a), len(b))
    if not scale or a == b:
        return 0.0
    prefix = 0
    while prefix < min(len(a), len(b)) and a[prefix] == b[prefix]:
        prefix += 1
    a, b = a[prefix:], b[prefix:]
    suffix = 0
    while suffix < min(len(a), len(b)) and a[-suffix - 1] == b[-suffix - 1]:
        suffix += 1
    if suffix:
        a, b = a[:-suffix], b[:-suffix]
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a) / scale
    equality = {}
    for i, char in enumerate(b):
        equality[char] = equality.get(char, 0) | (1 << i)
    positive = (1 << len(b)) - 1
    negative = 0
    distance = len(b)
    last = 1 << (len(b) - 1)
    for char in a:
        equal = equality.get(char, 0)
        diagonal = (((equal & positive) + positive) ^ positive) | equal | negative
        horizontal_positive = negative | ~(diagonal | positive)
        horizontal_negative = positive & diagonal
        distance += bool(horizontal_positive & last) - bool(horizontal_negative & last)
        horizontal_positive = (horizontal_positive << 1) | 1
        horizontal_negative <<= 1
        positive = horizontal_negative | ~(diagonal | horizontal_positive)
        negative = diagonal & horizontal_positive
    return distance / scale


def normalize_code_for_similarity(code: str) -> str:
    if not code:
        return ""
    lines: List[str] = []
    for ln in str(code).splitlines():
        s = ln.strip()
        if not s:
            continue
        lines.append(s)
    return "\n".join(lines).strip()


def text_similarity(a: str, b: str) -> float:
    na = normalize_code_for_similarity(a)
    nb = normalize_code_for_similarity(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return float(difflib.SequenceMatcher(a=na, b=nb).ratio())
