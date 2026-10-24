from typing import Iterable, List, Optional, Union


def generated_lengths(rows: Iterable[Iterable[int]], eos_token_id: Optional[Union[int, List[int]]]) -> List[int]:
    endings = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])
    endings.discard(None)
    lengths = []
    for row in rows:
        tokens = list(row)
        lengths.append(next((i + 1 for i, token in enumerate(tokens) if token in endings), len(tokens)))
    return lengths
