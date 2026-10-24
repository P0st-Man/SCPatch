from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Iterable, List, Optional
from src.tools.verification_scope import slither_relevant, include_internal_callees


def collect_slither_issues(
    slither: object, *, severities: Optional[Iterable[str]] = ("high", "medium"), scope=None,
) -> List[str]:
    if slither is None:
        raise ValueError("Slither analysis is unavailable")
    detectors = getattr(slither, "detectors", None) or []
    if not detectors:
        from slither.detectors import all_detectors
        from slither.detectors.abstract_detector import AbstractDetector
        classes = {
            cls for _, cls in inspect.getmembers(all_detectors, inspect.isclass)
            if cls is not AbstractDetector and issubclass(cls, AbstractDetector)
        }
        for cls in sorted(classes, key=lambda cls: cls.__name__):
            slither.register_detector(cls)
        detectors = getattr(slither, "detectors", None) or []
    if not detectors:
        raise RuntimeError("No Slither detectors were registered")
    include_internal_callees(slither, scope)
    results = slither.run_detectors()
    if not isinstance(results, (list, tuple)):
        raise RuntimeError("Slither returned no detector result collection")
    retained = None if severities is None else {str(s).strip().lower() for s in severities}
    issues: List[str] = []
    for batch in results:
        if not isinstance(batch, (list, tuple)):
            raise RuntimeError("Malformed Slither detector results")
        for item in batch:
            if not isinstance(item, Mapping):
                item = getattr(item, "data", None)
            if not isinstance(item, Mapping):
                raise RuntimeError("Malformed Slither finding")
            severity = str(item.get("impact") or item.get("severity") or "").lower().rsplit(".", 1)[-1]
            if severity not in {"high", "medium", "low", "informational", "optimization"}:
                raise RuntimeError("Slither finding has missing or unrecognized severity")
            if retained is not None and severity not in retained:
                continue
            if not slither_relevant(item, scope):
                continue
            issues.append(f"{item.get('check') or 'UnknownCheck'}: {item.get('description') or ''}")
    return issues
