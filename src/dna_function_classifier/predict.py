"""Predict one DNA sequence."""

import argparse, joblib
from .kmer import kmers_as_text
from .preprocessing import clean_sequence

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--sequence", required=True)
    args = p.parse_args()

    artifact = joblib.load(args.model)
    sequence = clean_sequence(args.sequence)
    document = kmers_as_text(sequence, int(artifact["config"]["kmer_size"]))
    print(artifact["model"].predict([document])[0])

if __name__ == "__main__":
    main()
