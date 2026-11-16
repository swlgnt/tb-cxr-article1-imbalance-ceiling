"""Pembuatan model CNN via timm (ResNet-18, ResNet-50, DenseNet201, Xception)."""
from __future__ import annotations

import timm
import torch.nn as nn


def build_model(timm_name: str, num_classes: int, pretrained: bool = True, drop_rate: float = 0.0):
    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
    data_cfg = timm.data.resolve_model_data_config(model)
    return model, data_cfg


def classifier_params(model: nn.Module):
    head = model.get_classifier()
    return list(head.parameters()) if isinstance(head, nn.Module) else []


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    head_ids = {id(p) for p in classifier_params(model)}
    for p in model.parameters():
        if id(p) not in head_ids:
            p.requires_grad = trainable


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    return {"params_total": total, "params_millions": round(total / 1e6, 3)}
