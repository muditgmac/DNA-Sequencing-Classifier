# Results

Training and evaluation scripts write their outputs to this directory.

A baseline training run produces:

```text
metrics.json
classification_report.csv
confusion_matrix.csv
confusion_matrix.png
```

`metrics.json` stores the main evaluation metrics and the model configuration used for the run.

The classification report contains per-class precision, recall, and F1-score. The confusion matrix is saved both as a CSV file and as an image.

Generated result files are ignored by default so that only verified experiment outputs are added to the repository. Results intended for publication in the README should be generated from the corresponding training or evaluation command and committed deliberately.
