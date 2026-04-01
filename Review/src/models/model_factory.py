from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Tuple
from urllib.error import URLError

import torch
import torch.nn as nn

from .efficientnet_b0 import build_efficientnet_b0
from .resnet50 import build_resnet50
from .vit_b_16 import build_vit_b_16


LOGGER = logging.getLogger(__name__)


ModelBuilder = Callable[[int, bool], Tuple[nn.Module, int]]


# Danh sách đăng ký các kiến trúc mô hình được hỗ trợ.
MODEL_BUILDERS: Dict[str, ModelBuilder] = {
    "resnet50": build_resnet50,
    "efficientnet_b0": build_efficientnet_b0,
    "vit_b_16": build_vit_b_16,
}


def _normalize_legacy_state_dict(model_name: str, state_dict: dict) -> dict:
    """Normalize known legacy key layouts before strict loading."""
    if model_name != "vit_b_16":
        return state_dict

    # Older ViT checkpoints used a direct Linear head at `heads.head.*`.
    if "heads.head.weight" in state_dict and "heads.head.1.weight" not in state_dict:
        state_dict["heads.head.1.weight"] = state_dict.pop("heads.head.weight")
    if "heads.head.bias" in state_dict and "heads.head.1.bias" not in state_dict:
        state_dict["heads.head.1.bias"] = state_dict.pop("heads.head.bias")
    return state_dict


def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    fallback_to_random: bool = False,
) -> Tuple[nn.Module, int]:
    # Cổng duy nhất để tạo mô hình cho cả huấn luyện, đánh giá và suy luận, đảm bảo tính nhất quán.
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unsupported model: {model_name}")

    builder = MODEL_BUILDERS[model_name]
    if pretrained:
        try:
            return builder(num_classes, True)
        except (OSError, ConnectionError, URLError) as exc:
            # Cơ chế dự phòng offline: nếu không tải được pretrained weights, sẽ tự động chuyển sang khởi tạo ngẫu nhiên để không ngắt quãng quá trình.
            LOGGER.warning(
                "Could not load pretrained weights for %s: %s",
                model_name,
                exc,
                exc_info=True,
            )
            if not fallback_to_random:
                raise
            LOGGER.warning("Falling back to random initialization for %s", model_name)
            return builder(num_classes, False)

    return builder(num_classes, False)


def load_checkpoint(
    model_name: str,
    num_classes: Optional[int],
    checkpoint_path: str,
    map_location: str | torch.device | None = None,
) -> nn.Module:
    # Hỗ trợ nạp cả 2 dạng file: chỉ chứa trọng số (state_dict) hoặc trọn bộ checkpoint (bao gồm cả optimizer, epoch...).
    state = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    state_dict = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format at: {checkpoint_path}")
    state_dict = _normalize_legacy_state_dict(model_name=model_name, state_dict=state_dict)

    resolved_num_classes = num_classes
    if isinstance(state, dict):
        ckpt_num_classes = state.get("num_classes")
        if isinstance(ckpt_num_classes, int) and ckpt_num_classes > 0:
            resolved_num_classes = ckpt_num_classes

    if resolved_num_classes is None:
        # Biện pháp cuối cùng: suy luận số lượng lớp (num_classes) dựa trên kích thước của lớp cuối cùng trong file trọng số.
        if model_name == "resnet50" and "fc.bias" in state_dict:
            resolved_num_classes = int(state_dict["fc.bias"].shape[0])
        elif model_name == "efficientnet_b0" and "classifier.1.bias" in state_dict:
            resolved_num_classes = int(state_dict["classifier.1.bias"].shape[0])
        elif model_name == "vit_b_16":
            if "heads.head.1.bias" in state_dict:
                resolved_num_classes = int(state_dict["heads.head.1.bias"].shape[0])
            elif "heads.head.bias" in state_dict:
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
    state = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if isinstance(state, dict) and isinstance(state.get("class_to_idx"), dict):
        return state["class_to_idx"]
    return None
