"""Fungsi loss: CE, weighted CE, focal loss."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_weights(labels: np.ndarray, n_classes: int, scheme: str = "inverse") -> torch.Tensor:
    cnt = np.bincount(labels, minlength=n_classes).astype(float)
    cnt[cnt == 0] = 1.0
    if scheme == "inverse":
        w = cnt.sum() / (n_classes * cnt)
    elif scheme == "sqrt_inverse":
        w = np.sqrt(cnt.sum() / (n_classes * cnt))
    elif scheme == "effective":  # Cui et al. 2019, beta=0.999
        beta = 0.999
        w = (1 - beta) / (1 - beta ** cnt)
        w = w / w.sum() * n_classes
    else:
        raise ValueError(scheme)
    return torch.tensor(w, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma, self.ls = gamma, label_smoothing
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none", label_smoothing=self.ls)
        pt = torch.exp(-F.cross_entropy(logits, target, reduction="none"))
        return ((1 - pt) ** self.gamma * ce).mean()


def build_loss(kind: str, train_labels: np.ndarray, n_classes: int, cfg: dict) -> nn.Module:
    ls = cfg.get("label_smoothing", 0.0)
    if kind == "ce":
        return nn.CrossEntropyLoss(label_smoothing=ls)
    w = class_weights(train_labels, n_classes, cfg.get("weight_scheme", "inverse"))
    if kind == "weighted_ce":
        return nn.CrossEntropyLoss(weight=w, label_smoothing=ls)
    if kind == "focal":
        return FocalLoss(cfg.get("focal_gamma", 2.0), None, ls)
    if kind == "weighted_focal":
        return FocalLoss(cfg.get("focal_gamma", 2.0), w, ls)
    raise ValueError(f"loss tidak dikenal: {kind}")
