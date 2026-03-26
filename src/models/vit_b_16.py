from __future__ import annotations

import torch.nn as nn
import torchvision.models as tv_models


FEATURE_DIM = 768


def build_vit_b_16(num_classes: int, pretrained: bool = True) -> tuple[nn.Module, int]:
    weights = tv_models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv_models.vit_b_16(weights=weights)
    model.heads.head = nn.Linear(FEATURE_DIM, num_classes)
    return model, FEATURE_DIM
