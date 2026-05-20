"""Clean and prepare the IMDb and Wikipedia movie genre datasets.

The script keeps IMDb and Wiki as two separate datasets, but maps both to the
same label space and writes comparable processed CSV files.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


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
    "science_fiction",
]

WIKI_LABEL_PRIORITY = [
    "horror",
    "science_fiction",
    "thriller",
    "adventure",
    "action",
    "crime",
    "romance",
    "comedy",
    "western",
    "drama",
]

WIKI_LABEL_PATTERNS = {
    "horror": [r"\bhorror\b"],
    "science_fiction": [
        r"\bscience[\s-]+fiction\b",
        r"\bsci[\s-]?fi\b",
        r"\bscifi\b",
    ],
    "thriller": [r"\bthriller\b", r"\bsuspense\b"],
    "action": [r"\baction\b"],
    "crime": [r"\bcrime\b", r"\bgangster\b", r"\bnoir\b"],
    "adventure": [r"\badventure\b"],
    "romance": [r"\bromance\b", r"\bromantic\b"],
    "comedy": [r"\bcomedy\b", r"\bcomic\b"],
    "western": [r"\bwestern\b"],
    "drama": [r"\bdrama\b", r"\bmelodrama\b", r"\bdramatic\b"],
}

OUTPUT_FIELDS = [
    "dataset",
    "sample_id",
    "title",
    "source_genre",
    "label",
    "text",
    "text_clean_tfidf",
    "text_clean_dl",
    "split",
]


@dataclass(frozen=True)
class PreparedRow:
    dataset: str
    sample_id: str
    title: str
    source_genre: str
    label: str
    text: str
    text_clean_tfidf: str
    text_clean_dl: str
    split: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare unified movie genre classification datasets."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing the data directory.",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=1000,
        help="Maximum samples kept per label for each dataset.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Stratified test ratio for each dataset.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=20,
        help="Drop samples whose raw text has fewer than this many words.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for balancing and train/test splitting.",
    )
    return parser.parse_args()


def require_nltk():
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import word_tokenize
    except ImportError as exc:
        raise SystemExit(
            "NLTK is required. Install it with:\n"
            "  uv add nltk\n"
            "Then download resources with:\n"
            "  uv run python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4"
        ) from exc

    resources = [
        (("tokenizers/punkt", "tokenizers/punkt.zip"), "punkt"),
        (("tokenizers/punkt_tab", "tokenizers/punkt_tab.zip"), "punkt_tab"),
        (("corpora/stopwords", "corpora/stopwords.zip"), "stopwords"),
        (("corpora/wordnet", "corpora/wordnet.zip"), "wordnet"),
        (("corpora/omw-1.4", "corpora/omw-1.4.zip"), "omw-1.4"),
    ]
    missing = []
    for paths, package in resources:
        found = False
        for path in paths:
            try:
                nltk.data.find(path)
                found = True
                break
            except LookupError:
                pass
        if not found:
            missing.append(package)

    if missing:
        packages = " ".join(missing)
        raise SystemExit(
            "Missing NLTK resources. Download them with:\n"
            f"  uv run python -m nltk.downloader {packages}"
        )

    return word_tokenize, set(stopwords.words("english")), WordNetLemmatizer()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_wiki_plot(text: str) -> str:
    text = re.sub(r"\[\d+\]", " ", text)
    text = re.sub(r"\[citation needed\]", " ", text, flags=re.IGNORECASE)
    return normalize_whitespace(text)


def word_count(text: str) -> int:
    return len(re.findall(r"\b[A-Za-z]+\b", text))


def map_imdb_genre(source_genre: str) -> str | None:
    genre = source_genre.strip().lower()
    if genre == "sci-fi":
        return "science_fiction"
    if genre in TARGET_LABELS:
        return genre
    return None


def map_wiki_genre(source_genre: str) -> str | None:
    genre = normalize_whitespace(source_genre.lower())
    if not genre or genre == "unknown":
        return None

    for label in WIKI_LABEL_PRIORITY:
        if any(re.search(pattern, genre) for pattern in WIKI_LABEL_PATTERNS[label]):
            return label
    return None


def clean_for_dl(text: str, word_tokenize) -> str:
    tokens = [
        token.lower()
        for token in word_tokenize(text)
        if re.search(r"[A-Za-z]", token)
    ]
    return " ".join(tokens)


def clean_for_tfidf(text: str, word_tokenize, stop_words: set[str], lemmatizer) -> str:
    tokens = []
    for token in word_tokenize(text.lower()):
        if not token.isalpha() or token in stop_words:
            continue
        lemma = lemmatizer.lemmatize(token)
        tokens.append(lemma)
    return " ".join(tokens)


def read_imdb_rows(
    imdb_dir: Path,
    word_tokenize,
    stop_words: set[str],
    lemmatizer,
    min_words: int,
) -> list[PreparedRow]:
    rows = []
    input_files = [
        imdb_dir / "train_data.txt",
        imdb_dir / "test_data_solution.txt",
    ]

    for input_file in input_files:
        with input_file.open("r", encoding="utf-8", errors="replace") as file:
            for line in file:
                parts = line.rstrip("\n").split(" ::: ", 3)
                if len(parts) != 4:
                    continue
                sample_id, title, source_genre, description = parts
                label = map_imdb_genre(source_genre)
                text = normalize_whitespace(description)
                if label is None or word_count(text) < min_words:
                    continue
                rows.append(
                    PreparedRow(
                        dataset="imdb",
                        sample_id=f"{input_file.stem}:{sample_id}",
                        title=title,
                        source_genre=source_genre,
                        label=label,
                        text=text,
                        text_clean_tfidf=clean_for_tfidf(
                            text, word_tokenize, stop_words, lemmatizer
                        ),
                        text_clean_dl=clean_for_dl(text, word_tokenize),
                    )
                )
    return rows


def read_wiki_rows(
    wiki_csv: Path,
    word_tokenize,
    stop_words: set[str],
    lemmatizer,
    min_words: int,
) -> list[PreparedRow]:
    rows = []
    with wiki_csv.open("r", encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.DictReader(file)
        for row_number, row in enumerate(reader, start=1):
            source_genre = row.get("Genre", "")
            label = map_wiki_genre(source_genre)
            text = clean_wiki_plot(row.get("Plot", ""))
            if label is None or word_count(text) < min_words:
                continue
            rows.append(
                PreparedRow(
                    dataset="wiki",
                    sample_id=str(row_number),
                    title=normalize_whitespace(row.get("Title", "")),
                    source_genre=source_genre,
                    label=label,
                    text=text,
                    text_clean_tfidf=clean_for_tfidf(
                        text, word_tokenize, stop_words, lemmatizer
                    ),
                    text_clean_dl=clean_for_dl(text, word_tokenize),
                )
            )
    return rows


def balance_rows(
    rows: Iterable[PreparedRow],
    max_per_label: int,
    seed: int,
) -> list[PreparedRow]:
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for row in rows:
        by_label[row.label].append(row)

    balanced = []
    for label in TARGET_LABELS:
        label_rows = by_label.get(label, [])
        rng.shuffle(label_rows)
        balanced.extend(label_rows[:max_per_label])
    rng.shuffle(balanced)
    return balanced


def stratified_split(
    rows: Iterable[PreparedRow],
    test_size: float,
    seed: int,
) -> list[PreparedRow]:
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for row in rows:
        by_label[row.label].append(row)

    split_rows = []
    for label_rows in by_label.values():
        rng.shuffle(label_rows)
        test_count = max(1, round(len(label_rows) * test_size))
        test_ids = {id(row) for row in label_rows[:test_count]}
        for row in label_rows:
            split = "test" if id(row) in test_ids else "train"
            split_rows.append(
                PreparedRow(
                    dataset=row.dataset,
                    sample_id=row.sample_id,
                    title=row.title,
                    source_genre=row.source_genre,
                    label=row.label,
                    text=row.text,
                    text_clean_tfidf=row.text_clean_tfidf,
                    text_clean_dl=row.text_clean_dl,
                    split=split,
                )
            )
    rng.shuffle(split_rows)
    return split_rows


def write_rows(rows: list[PreparedRow], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in OUTPUT_FIELDS})


def print_summary(name: str, rows: list[PreparedRow]) -> None:
    labels = Counter(row.label for row in rows)
    splits = Counter(row.split for row in rows)
    print(f"{name}: {len(rows)} rows")
    print(f"{name} label distribution: {dict(sorted(labels.items()))}")
    print(f"{name} split distribution: {dict(sorted(splits.items()))}")


def main() -> None:
    args = parse_args()
    word_tokenize, stop_words, lemmatizer = require_nltk()

    project_root = args.project_root
    imdb_dir = project_root / "data" / "Genre Classification Dataset"
    wiki_csv = project_root / "data" / "Wiki Movie Plots" / "wiki_movie_plots_deduped.csv"
    processed_dir = project_root / "data" / "processed"

    imdb_rows = read_imdb_rows(
        imdb_dir, word_tokenize, stop_words, lemmatizer, args.min_words
    )
    wiki_rows = read_wiki_rows(
        wiki_csv, word_tokenize, stop_words, lemmatizer, args.min_words
    )

    imdb_rows = stratified_split(
        balance_rows(imdb_rows, args.max_per_label, args.seed),
        args.test_size,
        args.seed,
    )
    wiki_rows = stratified_split(
        balance_rows(wiki_rows, args.max_per_label, args.seed),
        args.test_size,
        args.seed,
    )

    write_rows(imdb_rows, processed_dir / "imdb_cleaned.csv")
    write_rows(wiki_rows, processed_dir / "wiki_cleaned.csv")

    print_summary("IMDb", imdb_rows)
    print_summary("Wiki", wiki_rows)


if __name__ == "__main__":
    main()
