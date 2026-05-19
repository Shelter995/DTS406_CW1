"""Training and result-writing logic for traditional experiments."""

from __future__ import annotations

import csv
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report

from config import DATASETS, TARGET_LABELS
from dataset import read_rows, split_rows
from model import build_models, make_pipeline


def write_classification_report(
    output_path: Path,
    report: dict[str, dict[str, float] | float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["label", "precision", "recall", "f1-score", "support"]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for label, values in report.items():
            if isinstance(values, dict):
                writer.writerow(
                    {
                        "label": label,
                        "precision": values.get("precision", ""),
                        "recall": values.get("recall", ""),
                        "f1-score": values.get("f1-score", ""),
                        "support": values.get("support", ""),
                    }
                )
            else:
                writer.writerow(
                    {
                        "label": label,
                        "precision": "",
                        "recall": "",
                        "f1-score": values,
                        "support": "",
                    }
                )


def write_predictions(
    output_path: Path,
    sample_ids: list[str],
    titles: list[str],
    y_true: list[str],
    y_pred: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "title", "true_label", "predicted_label", "correct"]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for sample_id, title, true_label, predicted_label in zip(
            sample_ids, titles, y_true, y_pred, strict=True
        ):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "title": title,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "correct": true_label == predicted_label,
                }
            )


def summarize_metrics(
    dataset_name: str,
    model_name: str,
    y_true: list[str],
    y_pred: list[str],
    report: dict[str, dict[str, float] | float],
) -> dict[str, object]:
    macro = report["macro avg"]
    weighted = report["weighted avg"]
    if not isinstance(macro, dict) or not isinstance(weighted, dict):
        raise ValueError("classification_report returned unexpected summary format.")

    return {
        "dataset": dataset_name,
        "model": model_name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_precision": round(macro["precision"], 4),
        "macro_recall": round(macro["recall"], 4),
        "macro_f1": round(macro["f1-score"], 4),
        "weighted_precision": round(weighted["precision"], 4),
        "weighted_recall": round(weighted["recall"], 4),
        "weighted_f1": round(weighted["f1-score"], 4),
        "test_samples": len(y_true),
    }


def write_metrics(output_path: Path, metrics: list[dict[str, object]]) -> None:
    fields = [
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)


def run_experiments(project_root: Path, max_features: int, min_df: int) -> None:
    processed_dir = project_root / "data" / "processed"
    result_dir = project_root / "outputs" / "results" / "traditional"

    all_metrics = []
    for dataset_name, filename in DATASETS.items():
        rows = read_rows(processed_dir / filename)
        x_train, y_train, x_test, y_test, sample_ids, titles = split_rows(rows)

        print(
            f"{dataset_name}: train={len(y_train)}, test={len(y_test)}, "
            f"labels={len(set(y_train))}"
        )
        for model_name, model in build_models().items():
            pipeline = make_pipeline(model, max_features, min_df)
            pipeline.fit(x_train, y_train)
            y_pred = list(pipeline.predict(x_test))
            report = classification_report(
                y_test,
                y_pred,
                labels=TARGET_LABELS,
                output_dict=True,
                zero_division=0,
            )
            metrics = summarize_metrics(
                dataset_name, model_name, y_test, y_pred, report
            )
            all_metrics.append(metrics)

            write_classification_report(
                result_dir / f"{dataset_name}_{model_name}_classification_report.csv",
                report,
            )
            write_predictions(
                result_dir / f"{dataset_name}_{model_name}_predictions.csv",
                sample_ids,
                titles,
                y_test,
                y_pred,
            )
            print(
                f"  {model_name}: accuracy={metrics['accuracy']}, "
                f"macro_f1={metrics['macro_f1']}, "
                f"weighted_f1={metrics['weighted_f1']}"
            )

    write_metrics(result_dir / "traditional_metrics.csv", all_metrics)
    print(f"Wrote traditional model results to {result_dir}")
