"""Plot experiment figures from saved CSV outputs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


LABELS = [
    "action",
    "adventure",
    "comedy",
    "crime",
    "drama",
    "horror",
    "romance",
    "science_fiction",
    "thriller",
    "western",
]

MODEL_DISPLAY_NAMES = {
    "naive_bayes": "Naive Bayes",
    "linear_svm": "Linear SVM",
    "textcnn": "TextCNN",
    "tfidf_mlp": "TF-IDF MLP",
}

PREDICTION_FILES = {
    ("imdb", "naive_bayes"): Path("traditional/imdb_naive_bayes_predictions.csv"),
    ("imdb", "linear_svm"): Path("traditional/imdb_linear_svm_predictions.csv"),
    ("imdb", "textcnn"): Path("deep_learning/imdb_textcnn_predictions.csv"),
    ("imdb", "tfidf_mlp"): Path("deep_learning/imdb_tfidf_mlp_predictions.csv"),
    ("wiki", "naive_bayes"): Path("traditional/wiki_naive_bayes_predictions.csv"),
    ("wiki", "linear_svm"): Path("traditional/wiki_linear_svm_predictions.csv"),
    ("wiki", "textcnn"): Path("deep_learning/wiki_textcnn_predictions.csv"),
    ("wiki", "tfidf_mlp"): Path("deep_learning/wiki_tfidf_mlp_predictions.csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate confusion matrices and deep-learning training curves."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing outputs/results.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG output resolution.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def save_figure(fig: plt.Figure, output_path: Path, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    rows: list[dict[str, str]],
    dataset_name: str,
    model_name: str,
    output_dir: Path,
    dpi: int,
) -> None:
    y_true = [row["true_label"] for row in rows]
    y_pred = [row["predicted_label"] for row in rows]
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)

    fig, ax = plt.subplots(figsize=(9.5, 8))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(f"{dataset_name.upper()} - {MODEL_DISPLAY_NAMES[model_name]}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(LABELS)), labels=LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS)), labels=LABELS)

    threshold = matrix.max() * 0.55 if matrix.size else 0
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            text_color = "white" if value > threshold else "black"
            ax.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=7,
            )

    output_path = output_dir / "confusion_matrices" / (
        f"{dataset_name}_{model_name}_confusion_matrix.png"
    )
    save_figure(fig, output_path, dpi)


def plot_history_curve(
    history_rows: list[dict[str, str]],
    dataset_name: str,
    metric: str,
    ylabel: str,
    output_name: str,
    output_dir: Path,
    dpi: int,
) -> None:
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in history_rows:
        if row["dataset"] != dataset_name:
            continue
        grouped[row["model"]].append((int(row["epoch"]), float(row[metric])))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for model_name in ["textcnn", "tfidf_mlp"]:
        points = sorted(grouped.get(model_name, []))
        if not points:
            continue
        epochs = [epoch for epoch, _ in points]
        values = [value for _, value in points]
        ax.plot(
            epochs,
            values,
            marker="o",
            linewidth=2,
            markersize=4,
            label=MODEL_DISPLAY_NAMES[model_name],
        )

    ax.set_title(f"{dataset_name.upper()} - {ylabel}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    output_path = output_dir / "training_curves" / f"{dataset_name}_{output_name}.png"
    save_figure(fig, output_path, dpi)


def main() -> None:
    args = parse_args()
    project_root = args.project_root
    result_dir = project_root / "outputs" / "results"
    output_dir = project_root / "outputs" / "figures"

    for (dataset_name, model_name), relative_path in PREDICTION_FILES.items():
        prediction_path = result_dir / relative_path
        if not prediction_path.exists():
            raise SystemExit(f"Missing prediction file: {prediction_path}")
        plot_confusion_matrix(
            rows=read_csv(prediction_path),
            dataset_name=dataset_name,
            model_name=model_name,
            output_dir=output_dir,
            dpi=args.dpi,
        )

    history_path = result_dir / "deep_learning" / "training_history.csv"
    if not history_path.exists():
        raise SystemExit(f"Missing training history file: {history_path}")
    history_rows = read_csv(history_path)

    for dataset_name in ["imdb", "wiki"]:
        plot_history_curve(
            history_rows=history_rows,
            dataset_name=dataset_name,
            metric="train_loss",
            ylabel="Training Loss",
            output_name="training_loss",
            output_dir=output_dir,
            dpi=args.dpi,
        )
        plot_history_curve(
            history_rows=history_rows,
            dataset_name=dataset_name,
            metric="val_macro_f1",
            ylabel="Validation Macro-F1",
            output_name="validation_macro_f1",
            output_dir=output_dir,
            dpi=args.dpi,
        )

    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
