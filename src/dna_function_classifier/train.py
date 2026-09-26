"""Train the classical baseline."""

import argparse, json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from .dataset import load_many
from .modeling import BaselineConfig, build_baseline, evaluate_predictions, sequences_to_documents

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--sequence-column", default="sequence")
    p.add_argument("--target-column", default="class")
    p.add_argument("--kmer-size", type=int, default=4)
    p.add_argument("--ngram-size", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--model-out", default="models/baseline.joblib")
    p.add_argument("--results-dir", default="results")
    return p.parse_args()

def main():
    args = parse_args()
    frame = load_many(args.data, args.sequence_column, args.target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        frame["sequence"], frame["class"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=frame["class"],
    )

    cfg = BaselineConfig(args.kmer_size, args.ngram_size, args.alpha)
    model = build_baseline(cfg)
    model.fit(sequences_to_documents(X_train, cfg.kmer_size), y_train)
    pred = model.predict(sequences_to_documents(X_test, cfg.kmer_size))

    metrics, report, cm = evaluate_predictions(y_test, pred)
    metrics.update({
        "kmer_size": cfg.kmer_size,
        "ngram_size": cfg.ngram_size,
        "alpha": cfg.alpha,
        "test_size": args.test_size,
        "random_state": args.random_state,
    })

    model_out = Path(args.model_out)
    result_dir = Path(args.results_dir)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump({"model": model, "config": metrics}, model_out)
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    report.to_csv(result_dir / "classification_report.csv")
    cm.to_csv(result_dir / "confusion_matrix.csv")

    ConfusionMatrixDisplay(
        confusion_matrix=cm.to_numpy(),
        display_labels=[str(x).replace("true_", "") for x in cm.index],
    ).plot()
    plt.tight_layout()
    plt.savefig(result_dir / "confusion_matrix.png", dpi=180)
    plt.close()

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
