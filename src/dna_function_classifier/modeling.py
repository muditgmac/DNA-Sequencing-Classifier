"""Baseline model and metrics."""

from dataclasses import dataclass
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from .kmer import kmers_as_text

@dataclass(frozen=True)
class BaselineConfig:
    kmer_size: int = 4
    ngram_size: int = 1
    alpha: float = 0.1

def build_baseline(config: BaselineConfig) -> Pipeline:
    return Pipeline([
        ("vectorizer", CountVectorizer(
            analyzer="word",
            ngram_range=(config.ngram_size, config.ngram_size),
            lowercase=False,
            token_pattern=r"(?u)\b\w+\b",
        )),
        ("classifier", MultinomialNB(alpha=config.alpha)),
    ])

def sequences_to_documents(sequences, kmer_size: int):
    return sequences.map(lambda seq: kmers_as_text(seq, kmer_size))

def evaluate_predictions(y_true, y_pred):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n_samples": int(len(y_true)),
    }
    report = pd.DataFrame(
        classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    ).T
    labels = sorted(pd.Series(y_true).dropna().unique().tolist())
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    cm = pd.DataFrame(
        matrix,
        index=[f"true_{x}" for x in labels],
        columns=[f"pred_{x}" for x in labels],
    )
    return metrics, report, cm
