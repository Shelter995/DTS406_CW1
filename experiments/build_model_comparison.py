"""Build the main four-model comparison table for the report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


MAIN_MODELS = ["naive_bayes", "linear_svm", "textcnn", "tfidf_mlp"]
METRIC_FIELDS = [
    "dataset",
    "model",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "test_samples",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build main model comparison CSV.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser.parse_args()


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def main() -> None:
    args = parse_args()
    result_dir = args.project_root / "outputs" / "results"
    metric_paths = [
        result_dir / "traditional" / "traditional_metrics.csv",
        result_dir / "deep_learning" / "deep_learning_metrics.csv",
    ]

    rows = []
    for path in metric_paths:
        if not path.exists():
            raise SystemExit(f"Missing metrics file: {path}")
        rows.extend(read_metrics(path))

    filtered_rows = [row for row in rows if row["model"] in MAIN_MODELS]
    filtered_rows.sort(
        key=lambda row: (row["dataset"], MAIN_MODELS.index(row["model"]))
    )

    output_path = result_dir / "model_comparison.csv"
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(filtered_rows)

    print(f"Wrote main model comparison to {output_path}")


if __name__ == "__main__":
    main()
