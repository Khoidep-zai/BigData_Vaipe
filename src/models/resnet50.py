from __future__ import annotations

import torch.nn as nn
import torchvision.models as tv_models


FEATURE_DIM = 2048


def build_resnet50(num_classes: int, pretrained: bool = True) -> tuple[nn.Module, int]:
    weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = tv_models.resnet50(weights=weights)
    model.fc = nn.Linear(FEATURE_DIM, num_classes)
    return model, FEATURE_DIM
