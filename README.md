# DNA Sequence Function Classifier

I built this project to predict the functional class of a gene from the DNA sequence of its coding region using machine learning.

The baseline represents DNA sequences as overlapping k-mers and treats the resulting sequence of k-mers in a similar way to text. I use `CountVectorizer` to generate sparse count-based features and train a Multinomial Naive Bayes classifier on those features.

The project uses human, chimpanzee, and dog coding sequence datasets. The original notebook implementation achieved approximately 98.4% accuracy on its human test split.

I have since reorganized the project into a reproducible Python package, added separate training and evaluation scripts, added automated tests and GitHub Actions, and rerun the baseline using a stricter train-test workflow.

The current refactored implementation achieves:

- 97.95% accuracy on the held-out human test set
- 98.93% accuracy when the human-trained model is evaluated on chimpanzee sequences
- 91.71% accuracy when the same model is evaluated on dog sequences

I am currently extending the project with pretrained genomic transformer models and sequence-similarity-aware evaluation.

## Objective

The main objective is to test how much information about gene function can be learned directly from coding DNA sequence.

The project focuses on the following questions:

- Can DNA sequences be represented using methods commonly used in natural language processing?
- Can a classical machine learning model learn useful functional patterns from k-mer representations?
- How well does a model trained on human sequences transfer to sequences from other species?
- Does performance change when train-test separation is made stricter?
- Can pretrained genomic language models improve on the classical k-mer baseline?

The current datasets contain sequences from:

- Human
- Chimpanzee
- Dog

## Method

The current baseline follows this pipeline:

```text
DNA sequence
    |
    v
Sequence cleaning
    |
    v
Overlapping 6-mers
    |
    v
4-token n-gram representation
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

Each DNA sequence is cleaned and converted to uppercase.

The sequence is then divided into overlapping k-mers.

For example, consider:

```text
ATGCGTACG
```

Using `k = 4`, the overlapping k-mers would be:

```text
ATGC
TGCG
GCGT
CGTA
GTAC
TACG
```

The reproduced baseline uses 6-mers, matching the original notebook.

For a sequence of length `n` and k-mer size `k`, the number of overlapping k-mers is:

```text
n - k + 1
```

### 2. Feature generation

The generated k-mers are treated as tokens and converted into numerical features using `CountVectorizer`.

The original project used:

```python
CountVectorizer(ngram_range=(4, 4))
```

with DNA sequences first converted into overlapping 6-mers.

This means the classifier is not simply counting individual 6-mers. The vectorizer represents groups of four consecutive k-mer tokens.

The k-mer size and `CountVectorizer` n-gram size are kept as separate parameters in the refactored code so that they can be changed independently.

### 3. Train-test separation

The original notebook generated the complete vectorized human feature matrix before splitting it into training and test sets.

In the refactored pipeline, I split the human sequence dataset first and fit the vectorizer only on the training portion.

The held-out human sequences are transformed using the vocabulary learned from the training data.

The current pipeline uses:

```text
Test size: 20%
Random state: 42
Stratified split: Yes
```

This gives a cleaner estimate of performance because the feature extraction step is fitted only using the training data.

### 4. Classification

The baseline classifier is Multinomial Naive Bayes.

The reproduced configuration is:

```text
Model: Multinomial Naive Bayes
Alpha: 0.1
K-mer size: 6
CountVectorizer n-gram size: 4
Human test size: 20%
Random state: 42
```

Multinomial Naive Bayes works naturally with non-negative count-based features and provides a useful classical baseline before testing larger sequence models.

## Results

### Original notebook result

The original notebook reported:

```text
Accuracy:  98.4%
Precision: 98.4%
Recall:    98.4%
F1-score:  98.4%
```

These results came from the original implementation and train-test procedure.

### Refactored baseline

I reran the model using the original 6-mer representation, 4-token `CountVectorizer` n-grams, `alpha = 0.1`, and `random_state = 42`.

The refactored implementation performs the train-test split before fitting the vectorizer and uses a stratified split.

| Evaluation set | Accuracy | Macro F1 | Weighted F1 | Samples |
|---|---:|---:|---:|---:|
| Human held-out test set | 97.95% | 97.74% | 97.94% | 876 |
| Chimpanzee | 98.93% | 99.03% | 98.93% | 1,682 |
| Dog | 91.71% | 91.44% | 91.68% | 820 |

The human result is slightly lower than the 98.4% reported in the original notebook, but the refactored evaluation keeps the test set separate while fitting the feature representation.

### Cross-species evaluation

The classifier used for the cross-species experiment is trained using the human training set.

The fitted human model is then applied directly to the chimpanzee and dog datasets without retraining on either species.

```text
Human training data
        |
        v
6-mer generation
        |
        v
CountVectorizer
        |
        v
Multinomial Naive Bayes
        |
        +--------------------+
        |                    |
        v                    v
Chimpanzee evaluation    Dog evaluation
```

The chimpanzee dataset produced an accuracy of approximately 98.93%, while the dog dataset produced approximately 91.71%.

The lower performance on dog sequences provides a useful direction for further analysis of cross-species generalization and sequence similarity.

## Evaluation metrics

The refactored code calculates:

- Accuracy
- Macro F1-score
- Weighted F1-score
- Per-class precision
- Per-class recall
- Per-class F1-score
- Confusion matrix

Each training or evaluation run can save its results separately for later comparison.

## Repository structure

```text
DNA-Sequencing-Classifier/
|
|-- .github/
|   `-- workflows/
|       `-- tests.yml
|
|-- data/
|   `-- README.md
|
|-- models/
|   `-- .gitkeep
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
|-- DNA Sequencing and applying Classifier.ipynb
|-- human_data.txt
|-- chimp_data.txt
|-- dog_data.txt
|-- .gitignore
|-- LICENSE
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

The original notebook is kept in the repository as a record of the initial exploratory implementation.

The code under `src/` contains the refactored and reusable version of the pipeline.

## Installation

Python 3.10 or newer is recommended.

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the project and test dependencies:

```bash
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

Each dataset contains two columns:

```text
sequence
class
```

Example:

```text
sequence    class
ATGCCGTA... 0
GCTTAGCA... 3
```

The current repository contains:

```text
human_data.txt
chimp_data.txt
dog_data.txt
```

The files are tab-delimited.

The exact provenance and redistribution terms for the original datasets should be documented once the original dataset source is confirmed.

## Training the baseline

The reproduced baseline is trained on the human dataset.

Run:

```bash
python -m dna_function_classifier.train \
  --data human_data.txt \
  --kmer-size 6 \
  --ngram-size 4 \
  --alpha 0.1 \
  --test-size 0.20 \
  --random-state 42
```

The command uses:

```text
Human dataset
6-mer representation
4-token CountVectorizer n-grams
Multinomial Naive Bayes
alpha = 0.1
20% test split
random_state = 42
```

A successful training run generates:

```text
models/baseline.joblib

results/metrics.json
results/classification_report.csv
results/confusion_matrix.csv
results/confusion_matrix.png
```

Generated model and result files are excluded from version control by default.

## Evaluating the trained model

### Chimpanzee

After training the model on human sequences:

```bash
python -m dna_function_classifier.evaluate \
  --model models/baseline.joblib \
  --data chimp_data.txt \
  --results-dir results/chimp
```

The results are written under:

```text
results/chimp/
```

### Dog

Run:

```bash
python -m dna_function_classifier.evaluate \
  --model models/baseline.joblib \
  --data dog_data.txt \
  --results-dir results/dog
```

The results are written under:

```text
results/dog/
```

These evaluations use the same human-trained model and vectorizer.

The classifier is not retrained on the chimpanzee or dog datasets.

## Predicting a sequence

A saved model can also be used to classify an individual DNA sequence.

Example:

```bash
python -m dna_function_classifier.predict \
  --model models/baseline.joblib \
  --sequence "ATGCGTACGTTAGC"
```

The command loads the saved preprocessing and classification pipeline and returns the predicted functional class.

## Tests

Run the test suite with:

```bash
pytest
```

The current test suite covers:

- overlapping k-mer generation
- sequences shorter than the selected k-mer size
- k-mer text conversion
- invalid k-mer sizes
- sequence normalization
- whitespace removal
- ambiguous nucleotide handling
- invalid nucleotide handling
- empty sequence handling

The current suite contains eight automated tests.

## Continuous integration

The repository includes a GitHub Actions workflow under:

```text
.github/workflows/tests.yml
```

The workflow runs the test suite automatically on pushes and pull requests using supported Python environments.

This provides an additional check that the preprocessing and k-mer utilities continue to work after changes to the repository.

## Work in progress

### Genomic transformer classifier

The next major extension is a transformer-based genomic sequence classifier.

The current baseline relies on explicit k-mer counts. The transformer extension will instead use pretrained sequence representations that can incorporate surrounding sequence context.

The planned workflow is:

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

I plan to compare two settings.

#### Frozen embeddings

```text
DNA sequence
    |
    v
Pretrained genomic model
    |
    v
Fixed sequence embedding
    |
    v
Separate classifier
```

The pretrained model remains frozen and is used only to generate sequence representations.

#### Fine-tuned model

```text
DNA sequence
    |
    v
Pretrained genomic model
    |
    v
Task-specific classification head
    |
    v
End-to-end fine-tuning
```

This will allow a direct comparison between:

```text
Classical k-mer baseline
vs.
Frozen genomic transformer embeddings
vs.
Fine-tuned genomic transformer
```

No transformer performance values will be reported until the corresponding experiments have been completed.

### Sequence-similarity-aware evaluation

The cross-species baseline is now complete, but sequence similarity remains an important issue.

Closely related or duplicate sequences can make classification easier if similar sequences occur across training and evaluation sets.

The next evaluation stage will therefore include:

- exact duplicate detection
- duplicate removal across splits
- sequence similarity analysis
- similarity-aware train-test separation
- grouped evaluation where appropriate
- species-held-out testing under stricter similarity controls

The objective is to distinguish performance caused by transferable sequence patterns from performance that may depend heavily on highly related sequences.

### Additional classical baselines

I also plan to compare the Naive Bayes model against other classical classifiers using the same sequence representation.

Planned models include:

```text
Logistic Regression
Linear SVM
Multinomial Naive Bayes
```

Using the same input representation and evaluation splits will make the model comparison easier to interpret.

### Model interpretation

Another planned extension is to examine which sequence features influence individual predictions.

For classical models, this will include:

- class-associated k-mer frequencies
- discriminative k-mer analysis
- coefficient analysis for linear classifiers
- comparison of influential patterns across species

For transformer models, planned experiments include:

- token-level attribution
- sequence masking
- prediction changes after removing selected regions
- comparison of influential regions across species

## Current limitations

The current baseline has several limitations.

### Loss of positional information

The bag-of-words representation does not preserve the complete position of every sequence pattern.

Two sequences with similar k-mer composition can therefore receive similar representations even when the arrangement of those patterns differs.

### Limited long-range modeling

Count-based k-mer features do not directly model relationships between sequence regions that are separated by large distances.

This is one reason for testing transformer-based sequence representations.

### Sequence similarity

The current cross-species results do not yet control explicitly for highly similar or homologous sequences between datasets.

Similarity-aware evaluation is therefore required before drawing stronger conclusions about biological transfer between species.

### Dataset scope

The current experiments use only the human, chimpanzee, and dog datasets included in the original project.

Results should not be assumed to generalize to unrelated datasets, species, or functional annotation tasks without additional evaluation.

### Biological interpretation

The model identifies statistical patterns associated with the functional labels in the available dataset.

Its predictions should not be treated as experimentally validated biological annotations.

## Technologies used

Current implementation:

- Python
- pandas
- NumPy
- scikit-learn
- Jupyter Notebook
- CountVectorizer
- Multinomial Naive Bayes
- joblib
- matplotlib
- pytest
- GitHub Actions

In progress or planned:

- PyTorch
- Hugging Face Transformers
- pretrained genomic language models
- similarity-aware splitting
- additional classical classifiers
- sequence interpretation methods

## Method references

The references below are included for external methods, software, and pretrained models used or being evaluated in this project.

1. scikit-learn documentation. `CountVectorizer`, text feature extraction using token counts.

2. scikit-learn documentation. `MultinomialNB`, multinomial Naive Bayes classification for count-based features.

3. scikit-learn documentation. Common pitfalls and recommended practices, including train-test separation and prevention of preprocessing leakage.

4. Ji Y, Zhou Z, Liu H, Davuluri RV. DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome. *Bioinformatics*. 2021;37(15):2112-2120. DOI: 10.1093/bioinformatics/btab083.

5. Zhou Z, Ji Y, Li W, Dutta P, Davuluri R, Liu H. DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome. 2023. arXiv:2306.15006.

## License

This project is released under the GNU General Public License v3.0.

See `LICENSE` for the complete license terms.

## Disclaimer

This project is intended for machine learning and computational biology experimentation.

The predictions produced by the models are not experimentally validated biological annotations and should not be used for clinical or medical decisions.