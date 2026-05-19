"""Dataset and vocabulary helpers for deep learning experiments."""

from __future__ import annotations

import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from config import LABEL_TO_ID


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass(frozen=True)
class TextExample:
    sample_id: str
    title: str
    text: str
    label: str


class Vocabulary:
    def __init__(self, token_to_id: dict[str, int]) -> None:
        self.token_to_id = token_to_id
        self.pad_id = token_to_id[PAD_TOKEN]
        self.unk_id = token_to_id[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str, max_length: int) -> list[int]:
        token_ids = [
            self.token_to_id.get(token, self.unk_id)
            for token in text.split()[:max_length]
        ]
        if len(token_ids) < max_length:
            token_ids.extend([self.pad_id] * (max_length - len(token_ids)))
        return token_ids


class MovieTextDataset(Dataset):
    def __init__(
        self,
        examples: list[TextExample],
        vocab: Vocabulary,
        max_length: int,
    ) -> None:
        self.examples = examples
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        return {
            "input_ids": torch.tensor(
                self.vocab.encode(example.text, self.max_length),
                dtype=torch.long,
            ),
            "label": torch.tensor(LABEL_TO_ID[example.label], dtype=torch.long),
            "sample_id": example.sample_id,
            "title": example.title,
        }


class SparseTfidfDataset(Dataset):
    def __init__(self, features, examples: list[TextExample]) -> None:
        self.features = features
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        dense_features = self.features.getrow(index).toarray().ravel()
        return {
            "features": torch.tensor(dense_features, dtype=torch.float32),
            "label": torch.tensor(LABEL_TO_ID[example.label], dtype=torch.long),
            "sample_id": example.sample_id,
            "title": example.title,
        }


def normalize_title(title: str) -> str:
    return " ".join(token.lower() for token in title.replace("(", " ").replace(")", " ").split())


def read_examples(
    path: Path,
    text_field: str = "text_clean_dl",
    include_title: bool = False,
) -> tuple[list[TextExample], list[TextExample]]:
    train_examples = []
    test_examples = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            text = row[text_field]
            if include_title:
                text = f"{normalize_title(row['title'])} {text}"
            example = TextExample(
                sample_id=row["sample_id"],
                title=row["title"],
                text=text,
                label=row["label"],
            )
            if row["split"] == "train":
                train_examples.append(example)
            elif row["split"] == "test":
                test_examples.append(example)

    if not train_examples or not test_examples:
        raise ValueError("Both train and test examples are required.")
    return train_examples, test_examples


def read_tfidf_examples(path: Path) -> tuple[list[TextExample], list[TextExample]]:
    train_examples = []
    test_examples = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            example = TextExample(
                sample_id=row["sample_id"],
                title=row["title"],
                text=row["text_clean_tfidf"],
                label=row["label"],
            )
            if row["split"] == "train":
                train_examples.append(example)
            elif row["split"] == "test":
                test_examples.append(example)

    if not train_examples or not test_examples:
        raise ValueError("Both train and test examples are required.")
    return train_examples, test_examples


def stratified_validation_split(
    examples: list[TextExample],
    val_size: float,
    seed: int,
) -> tuple[list[TextExample], list[TextExample]]:
    rng = random.Random(seed)
    by_label: dict[str, list[TextExample]] = {}
    for example in examples:
        by_label.setdefault(example.label, []).append(example)

    train_examples = []
    val_examples = []
    for label_examples in by_label.values():
        rng.shuffle(label_examples)
        val_count = max(1, round(len(label_examples) * val_size))
        val_examples.extend(label_examples[:val_count])
        train_examples.extend(label_examples[val_count:])

    rng.shuffle(train_examples)
    rng.shuffle(val_examples)
    return train_examples, val_examples


def build_vocab(
    examples: list[TextExample],
    max_vocab_size: int,
    min_freq: int,
) -> Vocabulary:
    counter = Counter()
    for example in examples:
        counter.update(example.text.split())

    token_to_id = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }
    for token, count in counter.most_common(max_vocab_size - len(token_to_id)):
        if count < min_freq:
            break
        token_to_id[token] = len(token_to_id)
    return Vocabulary(token_to_id)
