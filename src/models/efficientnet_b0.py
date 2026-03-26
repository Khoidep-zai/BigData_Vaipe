from __future__ import annotations

import torch.nn as nn
import torchvision.models as tv_models


FEATURE_DIM = 1280


def build_efficientnet_b0(
    num_classes: int,
    pretrained: bool = True,
) -> tuple[nn.Module, int]:
    weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv_models.efficientnet_b0(weights=weights)
    model.classifier[-1] = nn.Linear(FEATURE_DIM, num_classes)
    return model, FEATURE_DIM
