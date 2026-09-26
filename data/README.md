# Data

The project uses coding DNA sequence datasets with a DNA sequence and its corresponding functional class.

The original experiments were performed on human, chimpanzee, and dog datasets.

## Expected columns

The default loader expects:

```text
sequence
class
```

`sequence` contains the coding DNA sequence.

`class` contains the target functional class.

CSV, TSV, and tab-delimited text files can be used. The delimiter is detected automatically by the loader.

## Recommended filenames

```text
human_data.txt
chimp_data.txt
dog_data.txt
```

The filenames are not fixed because dataset paths are passed directly to the training script.

## Dataset provenance

The source datasets are external to the code in this repository. Before redistributing the data files, the exact source, license, and any preprocessing performed on them should be documented here.

The code does not require the datasets to be stored in the repository. They can be kept locally under `data/` and passed to the training script by path.

## Evaluation

The current code supports combined training and testing. The next evaluation stage will add duplicate checks, similarity-aware splitting, and species-held-out experiments.
