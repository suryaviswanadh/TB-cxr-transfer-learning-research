from __future__ import annotations

from typing import Dict

import torch
from torchvision import models
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    ResNet50_Weights,
)

SUPPORTED_ARCHITECTURES = {
    "resnet50": "resnet50",
    "densenet121": "densenet121",
    "efficientnet_b0": "efficientnet_b0",
}


def create_model(architecture: str, num_classes: int = 2, pretrained: bool = True) -> torch.nn.Module:
    """Create a transfer-learning model for the tuberculosis chest X-ray task.

    Supported architectures:
    - resnet50
    - densenet121
    - efficientnet_b0
    """
    key = architecture.lower().strip()
    if key not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. Supported architectures: {sorted(SUPPORTED_ARCHITECTURES)}"
        )

    if key == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
        return model

    if key == "densenet121":
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, num_classes)
        return model

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    if hasattr(model, "classifier") and hasattr(model.classifier, "in_features"):
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, num_classes)
    else:
        in_features = model.classifier[-1].in_features
        model.classifier = torch.nn.Sequential(
            *list(model.classifier.children())[:-1],
            torch.nn.Linear(in_features, num_classes),
        )
    return model


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
