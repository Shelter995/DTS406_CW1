"""Generate basic statistics for processed movie genre datasets."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


DATASETS = {
    "imdb": "imdb_cleaned.csv",
    "wiki": "wiki_cleaned.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze processed IMDb and Wiki movie datasets."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing data/processed.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top words to export per dataset.",
    )
    return parser.parse_args()


def tokenize_clean_text(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z_]+\b", text.lower())


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def analyze_dataset(name: str, rows: list[dict[str, str]], top_n: int):
    label_counts = Counter(row["label"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)

    doc_lengths = []
    vocab = set()
    word_counts = Counter()
    for row in rows:
        tokens = tokenize_clean_text(row["text_clean_tfidf"])
        doc_lengths.append(len(tokens))
        vocab.update(tokens)
        word_counts.update(tokens)

    avg_length = round(sum(doc_lengths) / len(doc_lengths), 2) if doc_lengths else 0

    summary = {
        "dataset": name,
        "num_samples": len(rows),
        "num_labels": len(label_counts),
        "vocab_size": len(vocab),
        "avg_doc_length_tfidf_tokens": avg_length,
        "min_doc_length_tfidf_tokens": min(doc_lengths) if doc_lengths else 0,
        "max_doc_length_tfidf_tokens": max(doc_lengths) if doc_lengths else 0,
        "train_samples": split_counts.get("train", 0),
        "test_samples": split_counts.get("test", 0),
    }
    return summary, label_counts, word_counts.most_common(top_n)


def write_summary(output_dir: Path, summaries: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "num_samples",
        "num_labels",
        "vocab_size",
        "avg_doc_length_tfidf_tokens",
        "min_doc_length_tfidf_tokens",
        "max_doc_length_tfidf_tokens",
        "train_samples",
        "test_samples",
    ]
    with (output_dir / "dataset_statistics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)


def write_label_distribution(output_dir: Path, name: str, label_counts: Counter) -> None:
    with (output_dir / f"{name}_label_distribution.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["label", "count"])
        for label, count in sorted(label_counts.items()):
            writer.writerow([label, count])


def write_top_words(output_dir: Path, name: str, top_words: list[tuple[str, int]]) -> None:
    with (output_dir / f"{name}_top_words.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["word", "count"])
        writer.writerows(top_words)


def main() -> None:
    args = parse_args()
    processed_dir = args.project_root / "data" / "processed"
    output_dir = args.project_root / "outputs" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for name, filename in DATASETS.items():
        path = processed_dir / filename
        if not path.exists():
            raise SystemExit(
                f"Missing {path}. Run utils/preprocess_datasets.py first."
            )
        summary, label_counts, top_words = analyze_dataset(
            name, read_rows(path), args.top_n
        )
        summaries.append(summary)
        write_label_distribution(output_dir, name, label_counts)
        write_top_words(output_dir, name, top_words)
        print(f"{name}: analyzed {summary['num_samples']} rows")

    write_summary(output_dir, summaries)
    print(f"Wrote statistics to {output_dir}")


if __name__ == "__main__":
    main()
