import pytest
from dna_function_classifier.kmer import generate_kmers, kmers_as_text

def test_generate_overlapping_kmers():
    assert generate_kmers("ATGCGT", 4) == ["ATGC", "TGCG", "GCGT"]

def test_short_sequence():
    assert generate_kmers("ATG", 4) == []

def test_kmers_as_text():
    assert kmers_as_text("ATGCGT", 4) == "ATGC TGCG GCGT"

def test_invalid_k():
    with pytest.raises(ValueError):
        generate_kmers("ATGC", 0)
