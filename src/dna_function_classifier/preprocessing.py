"""DNA sequence preprocessing."""

import re

def clean_sequence(sequence: object, allow_n: bool = True) -> str:
    if sequence is None:
        raise ValueError("Sequence cannot be None.")
    cleaned = re.sub(r"\s+", "", str(sequence)).upper()
    if not cleaned:
        raise ValueError("Sequence cannot be empty.")
    allowed = r"^[ACGTN]+$" if allow_n else r"^[ACGT]+$"
    if re.fullmatch(allowed, cleaned) is None:
        raise ValueError("Sequence contains unsupported characters.")
    return cleaned
