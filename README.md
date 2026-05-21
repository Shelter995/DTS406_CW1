# DTS406 Movie Genre Topic Classification

## Project Overview

This project is a document topic classification coursework project. It classifies movie texts into unified movie genre labels using two datasets:

- **IMDb Genre Classification Dataset**: short movie plot summaries with promotional and emotional wording.
- Link:https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb
- **Wikipedia Movie Plots**: long movie plot descriptions with more factual and narrative writing.
- Link:https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots

Both datasets are cleaned and mapped into the same 10 labels:

```text
drama, comedy, horror, action, thriller, romance, western,
crime, adventure, science_fiction
```

The final four models are:

- Multinomial Naive Bayes + TF-IDF
- Linear SVM + TF-IDF
- TextCNN + word embeddings
- TF-IDF MLP neural classifier

## Project Structure

```text
DTS406/
├── data/
│   ├── Genre Classification Dataset/
│   ├── Wiki Movie Plots/
│   └── processed/
├── docs/
│   ├── main.tex
│   └── reference.bib
├── experiments/
│   ├── traditinal/
│   ├── deep_learning/
│   ├── build_model_comparison.py
│   └── plot_figures.py
├── outputs/
│   ├── figures/
│   ├── results/
│   └── tables/
├── utils/
│   ├── preprocess_datasets.py
│   └── analyze_processed_datasets.py
├── assignment.md
├── pyproject.toml
└── uv.lock
```

## Environment

The project uses Python 3.11 and `uv`.

Core dependencies:

- `nltk`
- `scikit-learn`
- `matplotlib`
- `torch==2.12.0+cu126`

Install or sync dependencies:

```powershell
uv sync
```

Download NLTK resources if needed:

```powershell
uv run python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4
```

## Data Preprocessing

Run preprocessing:

```powershell
uv run python utils\preprocess_datasets.py
```

This generates:

```text
data/processed/imdb_cleaned.csv
data/processed/wiki_cleaned.csv
```

Generate dataset statistics:

```powershell
uv run python utils\analyze_processed_datasets.py
```

This writes tables under:

```text
outputs/tables/
```

## Train Traditional Models

```powershell
uv run python experiments\traditinal\main.py
```

Outputs:

```text
outputs/results/traditional/
```

## Train Deep Learning Models

Final deep learning models are TextCNN and TF-IDF MLP.

```powershell
uv run python experiments\deep_learning\main.py --models textcnn,tfidf_mlp --sequence-field text_clean_tfidf --include-title --epochs 20 --batch-size 64 --embedding-dim 256 --num-filters 256 --filter-sizes 1,2,3,4,5 --mlp-hidden-dim 1024 --dropout 0.5 --learning-rate 0.0005 --weight-decay 0.0005 --label-smoothing 0.1 --patience 5 --tfidf-mode word_char --tfidf-max-features 80000 --tfidf-min-df 2
```

Outputs:

```text
outputs/results/deep_learning/
```

## Build Final Comparison Table

```powershell
uv run python experiments\build_model_comparison.py
```

Final comparison table:

```text
outputs/results/model_comparison.csv
```

## Generate Figures

```powershell
uv run python experiments\plot_figures.py
```

Figure outputs:

```text
outputs/figures/
```

## Current Results

| Dataset | Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---:|---:|---:|---:|
| IMDb | Naive Bayes | 0.5935 | 0.5952 | 0.5935 | 0.5864 |
| IMDb | Linear SVM | 0.5925 | 0.5870 | 0.5925 | 0.5878 |
| IMDb | TextCNN | 0.5410 | 0.5437 | 0.5410 | 0.5377 |
| IMDb | TF-IDF MLP | 0.5885 | 0.5956 | 0.5885 | 0.5849 |
| Wikipedia | Naive Bayes | 0.5757 | 0.5832 | 0.5776 | 0.5629 |
| Wikipedia | Linear SVM | 0.5836 | 0.5818 | 0.5889 | 0.5834 |
| Wikipedia | TextCNN | 0.5450 | 0.5703 | 0.5512 | 0.5491 |
| Wikipedia | TF-IDF MLP | 0.5872 | 0.5903 | 0.5930 | 0.5895 |

## Report

The LaTeX report is in:

```text
docs/main.tex
docs/reference.bib
```

If a TeX Live environment is available, compile with:

```powershell
cd docs
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```
