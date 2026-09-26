"""Evaluate a saved model on supplied data."""

import argparse, json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from .dataset import load_many
from .modeling import evaluate_predictions, sequences_to_documents

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--sequence-column", default="sequence")
    p.add_argument("--target-column", default="class")
    p.add_argument("--results-dir", default="results/evaluation")
    args = p.parse_args()

    artifact = joblib.load(args.model)
    model, config = artifact["model"], artifact["config"]
    frame = load_many(args.data, args.sequence_column, args.target_column)
    docs = sequences_to_documents(frame["sequence"], int(config["kmer_size"]))
    pred = model.predict(docs)
    metrics, report, cm = evaluate_predictions(frame["class"], pred)

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    report.to_csv(out / "classification_report.csv")
    cm.to_csv(out / "confusion_matrix.csv")
    ConfusionMatrixDisplay(
        confusion_matrix=cm.to_numpy(),
        display_labels=[str(x).replace("true_", "") for x in cm.index],
    ).plot()
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=180)
    plt.close()
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
