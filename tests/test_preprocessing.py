import pytest
from dna_function_classifier.preprocessing import clean_sequence

def test_clean_sequence():
    assert clean_sequence(" atgc\n gt ") == "ATGCGT"

def test_n_allowed():
    assert clean_sequence("ATNG") == "ATNG"

def test_invalid_character():
    with pytest.raises(ValueError):
        clean_sequence("ATGX")

def test_empty_sequence():
    with pytest.raises(ValueError):
        clean_sequence("   ")
