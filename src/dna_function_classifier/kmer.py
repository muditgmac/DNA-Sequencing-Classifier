"""K-mer utilities."""

def generate_kmers(sequence: str, k: int = 4) -> list[str]:
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if len(sequence) < k:
        return []
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def kmers_as_text(sequence: str, k: int = 4) -> str:
    return " ".join(generate_kmers(sequence, k))
