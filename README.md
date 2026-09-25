# DNA Sequence Function Classifier

I built this project to predict the functional class of a gene directly from the DNA sequence of its coding region using machine learning.

The baseline treats DNA as a sequence of overlapping k-mers. I convert the k-mers into a bag-of-words representation using `CountVectorizer` and train a Multinomial Naive Bayes classifier on the resulting sparse features. I tested the approach on human, chimpanzee, and dog sequence datasets.

The best configuration in the original experiments achieved approximately **98.4% accuracy** with a Multinomial Naive Bayes model using an alpha value of 0.1.

I am currently extending the project with a transformer-based genomic sequence classifier and a stricter evaluation setup for cross-species generalization and sequence similarity.

## Objective

The main objective is to test how much information about gene function can be learned from coding DNA sequence alone.

The project focuses on three questions:

- Can DNA sequences be represented using methods commonly used in natural language processing?
- Can a classical machine learning model learn useful functional patterns from k-mer representations?
- How well does the model generalize across sequences from different species?

The current experiments use sequence datasets from:

- Human
- Chimpanzee
- Dog

## Method

The current pipeline is:

```text
DNA sequence
    |
    v
Sequence cleaning
    |
    v
Overlapping k-mers
    |
    v
CountVectorizer
    |
    v
Multinomial Naive Bayes
    |
    v
Predicted functional class
```

### 1. Sequence preprocessing

Each DNA sequence is cleaned, converted to uppercase, and divided into overlapping k-mers.

For example, for the sequence:

```text
ATGCGTACG
```

using `k = 4`, the sequence becomes:

```text
ATGC
TGCG
GCGT
CGTA
GTAC
TACG
```

These k-mers are then treated as tokens.

### 2. Feature generation

I use `CountVectorizer` to convert the k-mer tokens into a sparse numerical representation. The resulting feature matrix records the occurrence of k-mer patterns in each sequence.

The k-mer length and the `CountVectorizer` n-gram size are kept as separate parameters so they can be tested independently.

### 3. Classification

The baseline classifier is Multinomial Naive Bayes. It was selected because it works naturally with non-negative count-based features and provides a simple baseline for sequence classification.

The best configuration in the original experiments used:

```text
Model: Multinomial Naive Bayes
Alpha: 0.1
Representation: k-mer bag-of-words
```

## Results

The best baseline result from the original experiments was:

```text
Accuracy: approximately 98.4%
```

The updated code also calculates:

- Accuracy
- Macro F1-score
- Weighted F1-score
- Per-class precision
- Per-class recall
- Per-class F1-score
- Confusion matrix

The next stage of the project will also include species-held-out evaluation so that the model is tested on a stricter generalization setting.

## Repository structure

```text
DNA-Sequence-Function-Classifier/
|
|-- data/
|   `-- README.md
|
|-- models/
|
|-- results/
|   `-- README.md
|
|-- src/
|   `-- dna_function_classifier/
|       |-- __init__.py
|       |-- dataset.py
|       |-- evaluate.py
|       |-- kmer.py
|       |-- modeling.py
|       |-- predict.py
|       |-- preprocessing.py
|       `-- train.py
|
|-- tests/
|   |-- test_kmer.py
|   `-- test_preprocessing.py
|
|-- .github/
|   `-- workflows/
|       `-- tests.yml
|
|-- .gitignore
|-- LICENSE
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

## Installation

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[test]"
```

On Windows:

```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[test]"
```

## Data format

The training script expects a sequence column and a target class column.

The default column names are:

```text
sequence
class
```

CSV, TSV, and tab-delimited text files are supported.

Example:

```text
sequence    class
ATGCCGTA... 0
GCTTAGCA... 3
```

The original experiments used human, chimpanzee, and dog DNA sequence datasets. Dataset provenance and licensing details should be documented in `data/README.md` before redistributing any source data.

## Training the baseline

Example using three datasets:

```bash
python -m dna_function_classifier.train \
  --data data/human_data.txt data/chimp_data.txt data/dog_data.txt \
  --kmer-size 4 \
  --ngram-size 1 \
  --alpha 0.1
```

If the experiment uses larger k-mers followed by word n-grams in `CountVectorizer`, the two values can be set independently. For example:

```bash
python -m dna_function_classifier.train \
  --data data/human_data.txt data/chimp_data.txt data/dog_data.txt \
  --kmer-size 6 \
  --ngram-size 4 \
  --alpha 0.1
```

A training run saves:

```text
models/baseline.joblib
results/metrics.json
results/classification_report.csv
results/confusion_matrix.csv
results/confusion_matrix.png
```

## Evaluating a saved model

A saved model can be evaluated on a separate dataset using:

```bash
python -m dna_function_classifier.evaluate \
  --model models/baseline.joblib \
  --data data/held_out_species.txt
```

This writes a new evaluation report under `results/evaluation/`.

## Predicting a sequence

```bash
python -m dna_function_classifier.predict \
  --model models/baseline.joblib \
  --sequence "ATGCGTACGTTAGC"
```

## Tests

Run the test suite with:

```bash
pytest
```

The current tests cover sequence cleaning and overlapping k-mer generation. Additional tests will be added as the transformer pipeline and similarity-aware evaluation are implemented.

## Work in progress

### Genomic transformer classifier

I am extending the project with pretrained genomic language models so that the classifier can use contextual sequence representations instead of relying only on k-mer counts.

The planned pipeline is:

```text
Raw DNA sequence
        |
        v
Genomic tokenizer
        |
        v
Pretrained genomic transformer
        |
        v
Sequence representation
        |
        v
Classification head
        |
        v
Predicted functional class
```

The first comparison will use the classical k-mer baseline against a pretrained genomic transformer. The transformer stage will be evaluated in two settings:

- frozen pretrained embeddings followed by a classifier
- end-to-end fine-tuning for sequence classification

The transformer results will only be reported after the experiments are completed.

### Cross-species generalization

I am also adding species-held-out experiments to test whether the model learns patterns that transfer across species instead of relying only on random splits from the same combined dataset.

Planned experiments include:

```text
Train: Human
Test: Chimpanzee, Dog
```

and:

```text
Train: Human + Chimpanzee
Test: Dog
```

### Sequence-similarity-aware evaluation

Closely related sequences can make a random train-test split easier than the actual generalization problem. The updated evaluation will therefore include:

- duplicate sequence detection
- duplicate removal across splits
- checks for highly similar sequences
- group-aware train-test splitting
- species-held-out evaluation

### Model interpretation

I also plan to add interpretation methods for both the classical and transformer models.

For the classical models, this will include:

- important k-mers for individual classes
- class-associated k-mer frequencies
- model coefficient analysis where applicable

For the transformer model, planned experiments include:

- token-level attribution
- sequence masking
- analysis of sequence regions that consistently affect a prediction

## Current limitations

The current baseline has several limitations:

- bag-of-words features discard much of the positional information in the sequence
- local k-mer counts do not directly model long-range sequence relationships
- random splitting can overestimate performance if similar sequences occur in both the training and test sets
- accuracy alone can hide weak performance on minority classes
- biological function cannot always be inferred from sequence alone

These limitations are the main reason for adding transformer-based representations and stricter evaluation.

## Technologies used

Current implementation:

- Python
- pandas
- NumPy
- scikit-learn
- Jupyter Notebook
- CountVectorizer
- Multinomial Naive Bayes
- pytest
- GitHub Actions

In progress:

- PyTorch
- Hugging Face Transformers
- pretrained genomic language models
- cross-species evaluation
- sequence-similarity-aware splitting
- sequence interpretation methods

## Method references

The external references below are included only for methods and models used or being evaluated in this project.

1. scikit-learn documentation, `CountVectorizer`: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
2. scikit-learn documentation, `MultinomialNB`: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
3. Ji Y, Zhou Z, Liu H, Davuluri RV. *DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome*. Bioinformatics. 2021;37(15):2112-2120. https://doi.org/10.1093/bioinformatics/btab083
4. Zhou Z, Ji Y, Li W, Dutta P, Davuluri R, Liu H. *DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome*. 2023. https://arxiv.org/abs/2306.15006

## License

This project is released under the MIT License. See `LICENSE` for details.

## Disclaimer

This project is intended for machine learning and computational biology experimentation. The predictions are not experimentally validated biological annotations and should not be used for clinical or medical decisions.
