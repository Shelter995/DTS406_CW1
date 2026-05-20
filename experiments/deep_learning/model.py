"""TextCNN and TF-IDF neural model definitions."""

from __future__ import annotations

import torch
from torch import nn


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        pad_id: int,
        embedding_dim: int = 128,
        num_filters: int = 128,
        filter_sizes: tuple[int, ...] = (3, 4, 5),
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.convs = nn.ModuleList(
            nn.Conv1d(embedding_dim, num_filters, kernel_size=size)
            for size in filter_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            features = torch.relu(conv(embedded))
            pooled.append(torch.max(features, dim=2).values)
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))


class TfidfMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 1024,
        dropout: float = 0.5,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        self.linear = nn.Linear(input_dim, num_classes)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features) + self.residual_scale * self.network(features)


class TfidfLinear(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


def build_model(
    model_name: str,
    vocab_size: int,
    num_classes: int,
    pad_id: int,
    embedding_dim: int,
    num_filters: int,
    filter_sizes: tuple[int, ...],
    dropout: float,
) -> nn.Module:
    if model_name == "textcnn":
        return TextCNN(
            vocab_size=vocab_size,
            num_classes=num_classes,
            pad_id=pad_id,
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            filter_sizes=filter_sizes,
            dropout=dropout,
        )
    raise ValueError(f"Unknown model: {model_name}")


def build_tfidf_mlp(
    input_dim: int,
    num_classes: int,
    hidden_dim: int,
    dropout: float,
    residual_scale: float = 1.0,
) -> nn.Module:
    return TfidfMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        residual_scale=residual_scale,
    )


def build_tfidf_linear(input_dim: int, num_classes: int) -> nn.Module:
    return TfidfLinear(input_dim=input_dim, num_classes=num_classes)
