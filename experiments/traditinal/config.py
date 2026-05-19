"""Configuration for traditional TF-IDF classification experiments."""

from __future__ import annotations

import argparse
from pathlib import Path


DATASETS = {
    "imdb": "imdb_cleaned.csv",
    "wiki": "wiki_cleaned.csv",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TF-IDF + traditional classifiers."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root containing data/processed.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=50000,
        help="Maximum TF-IDF vocabulary size.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Ignore terms appearing in fewer than this many documents.",
    )
    return parser.parse_args()
