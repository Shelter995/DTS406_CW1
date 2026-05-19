"""Configuration for TextCNN and GRU experiments."""

from __future__ import annotations

import argparse
from pathlib import Path


DATASETS = {
    "imdb": "imdb_cleaned.csv",
    "wiki": "wiki_cleaned.csv",
}

MAX_LENGTHS = {
    "imdb": 200,
    "wiki": 500,
}

TARGET_LABELS = [
    "drama",
    "comedy",
    "horror",
    "action",
    "thriller",
    "romance",
    "western",
    "crime",
    "adventure",
    "musical",
    "science_fiction",
]

LABEL_TO_ID = {label: index for index, label in enumerate(TARGET_LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TextCNN and BiLSTM movie genre classifiers."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root containing data/processed.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--filter-sizes", type=str, default="3,4,5")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--max-vocab-size", type=int, default=50000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--tfidf-max-features", type=int, default=50000)
    parser.add_argument("--tfidf-min-df", type=int, default=2)
    parser.add_argument(
        "--tfidf-mode",
        type=str,
        default="word",
        choices=["word", "word_char"],
    )
    parser.add_argument("--mlp-hidden-dim", type=int, default=1024)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument(
        "--sequence-field",
        type=str,
        default="text_clean_dl",
        choices=["text_clean_dl", "text_clean_tfidf"],
    )
    parser.add_argument("--include-title", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Train on the full training split for exactly --epochs epochs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        type=str,
        default="textcnn,tfidf_mlp",
        help="Comma-separated model names: textcnn,gru,bilstm,tfidf_mlp,tfidf_linear.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="imdb,wiki",
        help="Comma-separated dataset names: imdb,wiki.",
    )
    return parser.parse_args()
