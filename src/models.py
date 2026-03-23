from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, Optional, Tuple
from urllib.error import URLError

import torch
import torch.nn as nn
import torchvision.models as tv_models


LOGGER = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    name: str
    feature_dim: int


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "resnet50": ModelConfig(name="resnet50", feature_dim=2048),
    "efficientnet_b0": ModelConfig(name="efficientnet_b0", feature_dim=1280),
    "vit_b_16": ModelConfig(name="vit_b_16", feature_dim=768),
}


def _replace_classifier(
    backbone: nn.Module, in_features: int, num_classes: int
) -> nn.Module:
    backbone.fc = nn.Linear(in_features, num_classes)
    return backbone


def _replace_classifier_efficientnet(
    backbone: nn.Module, in_features: int, num_classes: int
) -> nn.Module:
    backbone.classifier[-1] = nn.Linear(in_features, num_classes)
    return backbone


def _replace_head_vit(
    backbone: nn.Module, in_features: int, num_classes: int
) -> nn.Module:
    backbone.heads.head = nn.Linear(in_features, num_classes)
    return backbone


def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    fallback_to_random: bool = False,
) -> Tuple[nn.Module, int]:
    """Create a classification model and return (model, feature_dim)."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}")

    cfg = MODEL_CONFIGS[model_name]

    def _build(weights_enabled: bool) -> nn.Module:
        if model_name == "resnet50":
            weights = (
                tv_models.ResNet50_Weights.IMAGENET1K_V2 if weights_enabled else None
            )
            backbone = tv_models.resnet50(weights=weights)
            return _replace_classifier(backbone, cfg.feature_dim, num_classes)
        if model_name == "efficientnet_b0":
            weights = (
                tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
                if weights_enabled
                else None
            )
            backbone = tv_models.efficientnet_b0(weights=weights)
            return _replace_classifier_efficientnet(
                backbone, cfg.feature_dim, num_classes
            )
        if model_name == "vit_b_16":
            weights = (
                tv_models.ViT_B_16_Weights.IMAGENET1K_V1 if weights_enabled else None
            )
            backbone = tv_models.vit_b_16(weights=weights)
            return _replace_head_vit(backbone, cfg.feature_dim, num_classes)
        raise ValueError(f"Unsupported model: {model_name}")

    if pretrained:
        try:
            model = _build(weights_enabled=True)
        except (OSError, ConnectionError, URLError) as exc:
            LOGGER.warning(
                "Could not load pretrained weights for %s: %s",
                model_name,
                exc,
                exc_info=True,
            )
            if fallback_to_random:
                LOGGER.warning("Falling back to random initialization for %s", model_name)
                model = _build(weights_enabled=False)
            else:
                raise
    else:
        model = _build(weights_enabled=False)

    return model, cfg.feature_dim


def load_checkpoint(
    model_name: str,
    num_classes: Optional[int],
    checkpoint_path: str,
    map_location: str | torch.device | None = None,
) -> nn.Module:
    """Load model weights from checkpoint.

    If checkpoint contains `num_classes`, it will be preferred.
    """
    state = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    state_dict = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state

    resolved_num_classes = num_classes
    if isinstance(state, dict):
        ckpt_num_classes = state.get("num_classes")
        if isinstance(ckpt_num_classes, int) and ckpt_num_classes > 0:
            resolved_num_classes = ckpt_num_classes

    if resolved_num_classes is None:
        if model_name == "resnet50" and "fc.bias" in state_dict:
            resolved_num_classes = int(state_dict["fc.bias"].shape[0])
        elif model_name == "efficientnet_b0" and "classifier.1.bias" in state_dict:
            resolved_num_classes = int(state_dict["classifier.1.bias"].shape[0])
        elif model_name == "vit_b_16" and "heads.head.bias" in state_dict:
            resolved_num_classes = int(state_dict["heads.head.bias"].shape[0])

    if resolved_num_classes is None:
        raise ValueError("Cannot resolve num_classes from checkpoint and input arguments.")

    model, _ = create_model(model_name, num_classes=resolved_num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    return model


def load_checkpoint_class_to_idx(
    checkpoint_path: str,
    map_location: str | torch.device | None = None,
) -> Optional[Dict[str, int]]:
    """Read class_to_idx metadata from checkpoint if available."""
    state = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if isinstance(state, dict) and isinstance(state.get("class_to_idx"), dict):
        return state["class_to_idx"]
    return None

