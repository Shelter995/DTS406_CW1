"""Dataset loading helpers for traditional experiments."""

from __future__ import annotations

import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def split_rows(rows: list[dict[str, str]]):
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    if not train_rows or not test_rows:
        raise ValueError("Both train and test rows are required.")

    x_train = [row["text_clean_tfidf"] for row in train_rows]
    y_train = [row["label"] for row in train_rows]
    x_test = [row["text_clean_tfidf"] for row in test_rows]
    y_test = [row["label"] for row in test_rows]
    test_ids = [row["sample_id"] for row in test_rows]
    titles = [row["title"] for row in test_rows]
    return x_train, y_train, x_test, y_test, test_ids, titles
