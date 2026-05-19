"""Training and evaluation logic for deep learning experiments."""

from __future__ import annotations

import csv
import copy
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from torch.utils.data import DataLoader

from config import DATASETS, ID_TO_LABEL, MAX_LENGTHS, TARGET_LABELS
from dataset import (
    MovieTextDataset,
    SparseTfidfDataset,
    build_vocab,
    read_examples,
    read_tfidf_examples,
    stratified_validation_split,
)
from model import build_model, build_tfidf_linear, build_tfidf_mlp


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(dataloader.dataset)


def train_tfidf_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        features = batch["features"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    y_true = []
    y_pred = []
    sample_ids = []
    titles = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        logits = model(input_ids)
        predictions = torch.argmax(logits, dim=1)

        y_true.extend(ID_TO_LABEL[index] for index in labels.cpu().tolist())
        y_pred.extend(ID_TO_LABEL[index] for index in predictions.cpu().tolist())
        sample_ids.extend(batch["sample_id"])
        titles.extend(batch["title"])
    return y_true, y_pred, sample_ids, titles


@torch.no_grad()
def evaluate_tfidf(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    y_true = []
    y_pred = []
    sample_ids = []
    titles = []
    for batch in dataloader:
        features = batch["features"].to(device)
        labels = batch["label"].to(device)
        logits = model(features)
        predictions = torch.argmax(logits, dim=1)

        y_true.extend(ID_TO_LABEL[index] for index in labels.cpu().tolist())
        y_pred.extend(ID_TO_LABEL[index] for index in predictions.cpu().tolist())
        sample_ids.extend(batch["sample_id"])
        titles.extend(batch["title"])
    return y_true, y_pred, sample_ids, titles


def get_macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    report = classification_report(
        y_true,
        y_pred,
        labels=TARGET_LABELS,
        output_dict=True,
        zero_division=0,
    )
    macro = report["macro avg"]
    if not isinstance(macro, dict):
        raise ValueError("classification_report returned unexpected format.")
    return float(macro["f1-score"])


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


def write_training_history(
    output_path: Path,
    history: list[dict[str, object]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "model", "epoch", "train_loss", "val_macro_f1"]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def run_experiments(args) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processed_dir = args.project_root / "data" / "processed"
    result_dir = args.project_root / "outputs" / "results" / "deep_learning"
    selected_datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    selected_models = [name.strip() for name in args.models.split(",") if name.strip()]
    filter_sizes = tuple(
        int(size.strip()) for size in args.filter_sizes.split(",") if size.strip()
    )

    all_metrics = []
    all_history = []
    for dataset_name in selected_datasets:
        filename = DATASETS[dataset_name]
        full_train_examples, test_examples = read_examples(
            processed_dir / filename,
            text_field=args.sequence_field,
            include_title=args.include_title,
        )
        if args.no_validation:
            train_examples = full_train_examples
            val_examples = []
        else:
            train_examples, val_examples = stratified_validation_split(
                full_train_examples,
                args.val_size,
                args.seed,
            )
        max_length = MAX_LENGTHS[dataset_name]
        vocab = build_vocab(train_examples, args.max_vocab_size, args.min_freq)
        train_dataset = MovieTextDataset(train_examples, vocab, max_length)
        val_dataset = MovieTextDataset(val_examples, vocab, max_length) if val_examples else None
        test_dataset = MovieTextDataset(test_examples, vocab, max_length)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
            )
            if val_dataset is not None
            else None
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        print(
            f"{dataset_name}: train={len(train_dataset)}, val={len(val_dataset) if val_dataset is not None else 0}, "
            f"test={len(test_dataset)}, vocab={len(vocab)}, max_len={max_length}"
        )
        for model_name in selected_models:
            if model_name in {"tfidf_mlp", "tfidf_linear"}:
                metrics, history = run_tfidf_mlp(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    data_path=processed_dir / filename,
                    result_dir=result_dir,
                    args=args,
                    device=device,
                )
                all_metrics.append(metrics)
                all_history.extend(history)
                continue

            set_seed(args.seed)
            model = build_model(
                model_name=model_name,
                vocab_size=len(vocab),
                num_classes=len(TARGET_LABELS),
                pad_id=vocab.pad_id,
                embedding_dim=args.embedding_dim,
                hidden_dim=args.hidden_dim,
                num_filters=args.num_filters,
                filter_sizes=filter_sizes,
                dropout=args.dropout,
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
            best_val_f1 = -1.0
            best_state = None
            epochs_without_improvement = 0

            for epoch in range(1, args.epochs + 1):
                loss = train_one_epoch(
                    model=model,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    device=device,
                )
                val_macro_f1 = 0.0
                if val_loader is not None:
                    val_true, val_pred, _, _ = evaluate(model, val_loader, device)
                    val_macro_f1 = get_macro_f1(val_true, val_pred)
                all_history.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "epoch": epoch,
                        "train_loss": round(loss, 6),
                        "val_macro_f1": round(val_macro_f1, 6),
                    }
                )
                if val_loader is None:
                    print(f"  {model_name} epoch {epoch}/{args.epochs}: loss={loss:.4f}")
                else:
                    print(
                        f"  {model_name} epoch {epoch}/{args.epochs}: "
                        f"loss={loss:.4f}, val_macro_f1={val_macro_f1:.4f}"
                    )

                    if val_macro_f1 > best_val_f1:
                        best_val_f1 = val_macro_f1
                        best_state = copy.deepcopy(model.state_dict())
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                        if epochs_without_improvement >= args.patience:
                            print(
                                f"  {model_name}: early stopping after epoch {epoch}; "
                                f"best_val_macro_f1={best_val_f1:.4f}"
                            )
                            break

            if best_state is not None:
                model.load_state_dict(best_state)

            y_true, y_pred, sample_ids, titles = evaluate(model, test_loader, device)
            report = classification_report(
                y_true,
                y_pred,
                labels=TARGET_LABELS,
                output_dict=True,
                zero_division=0,
            )
            metrics = summarize_metrics(dataset_name, model_name, y_true, y_pred, report)
            all_metrics.append(metrics)

            write_classification_report(
                result_dir / f"{dataset_name}_{model_name}_classification_report.csv",
                report,
            )
            write_predictions(
                result_dir / f"{dataset_name}_{model_name}_predictions.csv",
                sample_ids,
                titles,
                y_true,
                y_pred,
            )
            print(
                f"  {model_name}: accuracy={metrics['accuracy']}, "
                f"macro_f1={metrics['macro_f1']}, "
                f"weighted_f1={metrics['weighted_f1']}"
            )

    write_metrics(result_dir / "deep_learning_metrics.csv", all_metrics)
    write_training_history(result_dir / "training_history.csv", all_history)
    print(f"Wrote deep learning results to {result_dir}")


def run_tfidf_mlp(
    model_name: str,
    dataset_name: str,
    data_path: Path,
    result_dir: Path,
    args,
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    full_train_examples, test_examples = read_tfidf_examples(data_path)
    if args.no_validation:
        train_examples = full_train_examples
        val_examples = []
    else:
        train_examples, val_examples = stratified_validation_split(
            full_train_examples,
            args.val_size,
            args.seed,
        )
    if args.tfidf_mode == "word_char":
        word_features = max(1, int(args.tfidf_max_features * 0.7))
        char_features = max(1, args.tfidf_max_features - word_features)
        vectorizer = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        lowercase=False,
                        token_pattern=r"(?u)\b\w+\b",
                        ngram_range=(1, 2),
                        min_df=args.tfidf_min_df,
                        max_df=0.95,
                        max_features=word_features,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=args.tfidf_min_df,
                        max_features=char_features,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
    else:
        vectorizer = TfidfVectorizer(
            lowercase=False,
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
            min_df=args.tfidf_min_df,
            max_df=0.95,
            max_features=args.tfidf_max_features,
            sublinear_tf=True,
        )
    train_texts = [example.text for example in train_examples]
    val_texts = [example.text for example in val_examples]
    test_texts = [example.text for example in test_examples]
    x_train = vectorizer.fit_transform(train_texts)
    x_val = vectorizer.transform(val_texts) if val_texts else None
    x_test = vectorizer.transform(test_texts)

    train_dataset = SparseTfidfDataset(x_train, train_examples)
    val_dataset = SparseTfidfDataset(x_val, val_examples) if x_val is not None else None
    test_dataset = SparseTfidfDataset(x_test, test_examples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        if val_dataset is not None
        else None
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    set_seed(args.seed)
    if model_name == "tfidf_linear":
        model = build_tfidf_linear(
            input_dim=x_train.shape[1],
            num_classes=len(TARGET_LABELS),
        ).to(device)
    else:
        model = build_tfidf_mlp(
            input_dim=x_train.shape[1],
            num_classes=len(TARGET_LABELS),
            hidden_dim=args.mlp_hidden_dim,
            dropout=args.dropout,
            residual_scale=args.residual_scale,
        ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    best_val_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    history = []

    print(
        f"{dataset_name} {model_name}: train={len(train_dataset)}, "
        f"val={len(val_dataset) if val_dataset is not None else 0}, test={len(test_dataset)}, "
        f"features={x_train.shape[1]}"
    )
    for epoch in range(1, args.epochs + 1):
        loss = train_tfidf_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        val_macro_f1 = 0.0
        if val_loader is not None:
            val_true, val_pred, _, _ = evaluate_tfidf(model, val_loader, device)
            val_macro_f1 = get_macro_f1(val_true, val_pred)
        history.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "epoch": epoch,
                "train_loss": round(loss, 6),
                "val_macro_f1": round(val_macro_f1, 6),
            }
        )
        if val_loader is None:
            print(f"  {model_name} epoch {epoch}/{args.epochs}: loss={loss:.4f}")
        else:
            print(
                f"  {model_name} epoch {epoch}/{args.epochs}: "
                f"loss={loss:.4f}, val_macro_f1={val_macro_f1:.4f}"
            )

            if val_macro_f1 > best_val_f1:
                best_val_f1 = val_macro_f1
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    print(
                    f"  {model_name}: early stopping after epoch {epoch}; "
                    f"best_val_macro_f1={best_val_f1:.4f}"
                )
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_pred, sample_ids, titles = evaluate_tfidf(model, test_loader, device)
    report = classification_report(
        y_true,
        y_pred,
        labels=TARGET_LABELS,
        output_dict=True,
        zero_division=0,
    )
    metrics = summarize_metrics(dataset_name, model_name, y_true, y_pred, report)
    write_classification_report(
        result_dir / f"{dataset_name}_{model_name}_classification_report.csv",
        report,
    )
    write_predictions(
        result_dir / f"{dataset_name}_{model_name}_predictions.csv",
        sample_ids,
        titles,
        y_true,
        y_pred,
    )
    print(
        f"  {model_name}: accuracy={metrics['accuracy']}, "
        f"macro_f1={metrics['macro_f1']}, weighted_f1={metrics['weighted_f1']}"
    )
    return metrics, history
