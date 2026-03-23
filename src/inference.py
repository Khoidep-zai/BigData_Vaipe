from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import build_transforms, compute_image_statistics, pil_loader, focus_on_object
from .metadata import MedicineMetadataIndex
from .models import load_checkpoint, load_checkpoint_class_to_idx


@dataclass
class ComparisonResult:
    predicted_class: str
    similarity_score: float
    color_score: float
    size_score: float
    texture_score: float
    num_true_features: int
    is_true: bool
    details: Dict[str, object]


# ---------------------------------------------------------------------------
# Cache toàn cục – giúp tránh load/tạo lại các đối tượng nặng
# ---------------------------------------------------------------------------
_MODEL_CACHE: Dict[Tuple[str, str, str], Tuple[nn.Module, Dict[int, str]]] = {}
_EVAL_TRANSFORM = None  # cache build_transforms(train=False)


def _get_eval_transform():
    """Trả về transform cho inference (cached, chỉ tạo 1 lần)."""
    global _EVAL_TRANSFORM
    if _EVAL_TRANSFORM is None:
        _EVAL_TRANSFORM = build_transforms(train=False)
    return _EVAL_TRANSFORM


def _resolve_device(device_str: str | None) -> torch.device:
    """Chọn device, tự fallback sang CPU nếu CUDA không khả dụng."""
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA không khả dụng, tự động chuyển sang CPU.")
        return torch.device("cpu")
    return torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))


def _get_or_load_model(
    model_name: str,
    class_to_idx: Dict[str, int],
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[nn.Module, Dict[int, str]]:
    key = (model_name, checkpoint_path, str(device))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    ckpt_class_to_idx = load_checkpoint_class_to_idx(checkpoint_path, map_location=device)
    num_classes = len(class_to_idx)
    if ckpt_class_to_idx:
        num_classes = len(ckpt_class_to_idx)

    model = load_checkpoint(
        model_name,
        num_classes=num_classes,
        checkpoint_path=checkpoint_path,
        map_location=device,
    ).to(device)
    model.eval()

    if ckpt_class_to_idx:
        inv_class_to_idx = {v: k for k, v in ckpt_class_to_idx.items()}
    else:
        inv_class_to_idx = {v: k for k, v in class_to_idx.items()}

    _MODEL_CACHE[key] = (model, inv_class_to_idx)
    return model, inv_class_to_idx


# ---------------------------------------------------------------------------
# Feature extraction via forward-hook – an toàn & hiệu năng hơn cắt module
# ---------------------------------------------------------------------------

def _extract_features(
    model: nn.Module, image_tensor: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Trích đặc trưng penultimate-layer bằng forward hook (an toàn cho mọi kiến trúc)."""
    model.eval()
    image_tensor = image_tensor.to(device)

    features: List[torch.Tensor] = []

    def _hook_fn(_module, _input, output):
        features.append(output)

    # Xác định layer cuối cùng trước classifier cho từng kiến trúc
    hook_handle = None
    if hasattr(model, "fc"):
        # ResNet-style: lấy output từ avgpool (trước fc)
        hook_handle = model.avgpool.register_forward_hook(_hook_fn)
    elif hasattr(model, "classifier"):
        # EfficientNet-style: lấy output từ avgpool (trước classifier)
        hook_handle = model.avgpool.register_forward_hook(_hook_fn)
    elif hasattr(model, "heads"):
        # ViT-style: lấy output từ encoder (trước heads)
        hook_handle = model.encoder.register_forward_hook(_hook_fn)
    else:
        # Fallback: dùng logits làm features
        with torch.no_grad():
            use_amp = device.type == "cuda"
            with torch.amp.autocast("cuda", enabled=use_amp):
                feat = model(image_tensor)
            feat = F.normalize(feat, p=2, dim=1)
            return feat.cpu()

    # Chạy forward để hook thu thập features
    with torch.no_grad():
        use_amp = device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp):
            _ = model(image_tensor)

    hook_handle.remove()

    if not features:
        raise RuntimeError("Forward hook không thu được features.")

    feat = features[0]
    feat = torch.flatten(feat, 1)
    feat = F.normalize(feat, p=2, dim=1)
    return feat.cpu()


def _forward_logits_and_features(
    model: nn.Module, image_tensor: torch.Tensor, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run one forward pass and return both logits and normalized features.

    This avoids running the same image multiple times when we need both
    predicted class and feature vector.
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    features: List[torch.Tensor] = []

    def _hook_fn(_module, _input, output):
        features.append(output)

    hook_handle = None
    if hasattr(model, "fc"):
        hook_handle = model.avgpool.register_forward_hook(_hook_fn)
    elif hasattr(model, "classifier"):
        hook_handle = model.avgpool.register_forward_hook(_hook_fn)
    elif hasattr(model, "heads"):
        hook_handle = model.encoder.register_forward_hook(_hook_fn)

    with torch.no_grad():
        use_amp = device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(image_tensor)

    if hook_handle is not None:
        hook_handle.remove()

    if features:
        feat = torch.flatten(features[0], 1)
    else:
        feat = logits

    feat = F.normalize(feat, p=2, dim=1)
    return logits.cpu(), feat.cpu()


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a, b).item())


def _compare_colors(img_a: Image.Image, img_b: Image.Image, threshold: float = 0.15) -> float:
    stats_a = compute_image_statistics(img_a)
    stats_b = compute_image_statistics(img_b)
    diff = math.sqrt(
        (stats_a["mean_r"] - stats_b["mean_r"]) ** 2
        + (stats_a["mean_g"] - stats_b["mean_g"]) ** 2
        + (stats_a["mean_b"] - stats_b["mean_b"]) ** 2
    )
    score = max(0.0, 1.0 - diff / threshold)
    return score


def _compare_size(img_a: Image.Image, img_b: Image.Image, threshold: float = 0.3) -> float:
    stats_a = compute_image_statistics(img_a)
    stats_b = compute_image_statistics(img_b)
    diff_ar = abs(stats_a["aspect_ratio"] - stats_b["aspect_ratio"])
    score = max(0.0, 1.0 - diff_ar / threshold)
    return score


def _compare_texture(img_a: Image.Image, img_b: Image.Image, threshold: float = 0.2) -> float:
    # Simple proxy: standard deviation of grayscale values
    gray_a = np.array(img_a.convert("L"), dtype=np.float32) / 255.0
    gray_b = np.array(img_b.convert("L"), dtype=np.float32) / 255.0
    std_a = float(gray_a.std())
    std_b = float(gray_b.std())
    diff = abs(std_a - std_b)
    score = max(0.0, 1.0 - diff / threshold)
    return score


def _count_true_features(
    sim: float, color: float, size: float, texture: float,
    sim_thresh: float, color_thresh: float, size_thresh: float, texture_thresh: float,
) -> int:
    count = 0
    if sim >= sim_thresh:
        count += 1
    if color >= color_thresh:
        count += 1
    if size >= size_thresh:
        count += 1
    if texture >= texture_thresh:
        count += 1
    return count


def compare_pill_images(
    model_name: str,
    checkpoint_path: str,
    class_to_idx: Dict[str, int],
    sample_image_path: str,
    query_image_path: str,
    device_str: str | None = None,
    expected_class_name: Optional[str] = None,
    metadata_index: Optional[MedicineMetadataIndex] = None,
    similarity_threshold: float = 0.7,
    color_threshold: float = 0.6,
    size_threshold: float = 0.6,
    texture_threshold: float = 0.6,
    min_true_features: int = 3,
) -> ComparisonResult:
    """Compare a sample pill image and a query image and decide True/False."""
    device = _resolve_device(device_str)

    # Load model (có cache để tránh load nhiều lần)
    model, inv_class_to_idx = _get_or_load_model(
        model_name=model_name,
        class_to_idx=class_to_idx,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # Load and transform images
    transform = _get_eval_transform()
    sample_img_pil = focus_on_object(pil_loader(sample_image_path), scale=0.85)
    query_img_pil = focus_on_object(pil_loader(query_image_path), scale=0.85)

    sample_tensor = transform(sample_img_pil).unsqueeze(0)
    query_tensor = transform(query_img_pil).unsqueeze(0)

    # Predict class + query features in one forward pass
    query_logits, feat_query = _forward_logits_and_features(model, query_tensor, device)
    pred_idx = int(torch.argmax(query_logits, dim=1).item())

    predicted_class = inv_class_to_idx.get(pred_idx, f"class_{pred_idx}")

    # Feature similarity
    feat_sample = _extract_features(model, sample_tensor, device)
    sim_score = _cosine_similarity(feat_sample, feat_query)

    # Color, size, texture comparisons (mapped to 0-1 scores)
    color_score = _compare_colors(sample_img_pil, query_img_pil)
    size_score = _compare_size(sample_img_pil, query_img_pil)
    texture_score = _compare_texture(sample_img_pil, query_img_pil)

    true_features = _count_true_features(
        sim_score, color_score, size_score, texture_score,
        similarity_threshold, color_threshold, size_threshold, texture_threshold,
    )

    class_match = True
    if expected_class_name:
        class_match = predicted_class == expected_class_name

    metadata_expected = metadata_index.best_match(expected_class_name) if (metadata_index and expected_class_name) else None
    metadata_pred = metadata_index.best_match(predicted_class) if metadata_index else None
    semantic_group_match = True
    if metadata_expected and metadata_pred:
        if metadata_expected.active_group and metadata_pred.active_group:
            semantic_group_match = metadata_expected.active_group == metadata_pred.active_group

    semantic_penalty = 1 if not semantic_group_match else 0
    effective_true_features = max(0, true_features - semantic_penalty)
    is_true = class_match and (effective_true_features >= min_true_features)

    details = {
        "similarity_score": sim_score,
        "color_score": color_score,
        "size_score": size_score,
        "texture_score": texture_score,
        "class_match": float(1 if class_match else 0),
        "semantic_group_match": float(1 if semantic_group_match else 0),
        "effective_true_features": float(effective_true_features),
    }
    if metadata_expected is not None:
        details["expected_active_group"] = metadata_expected.active_group
        details["expected_disease_vi"] = metadata_expected.disease_vi
        details["expected_medicine_name"] = metadata_expected.medicine_name
    if metadata_pred is not None:
        details["pred_active_group"] = metadata_pred.active_group
        details["pred_disease_vi"] = metadata_pred.disease_vi
        details["pred_medicine_name"] = metadata_pred.medicine_name

    return ComparisonResult(
        predicted_class=predicted_class,
        similarity_score=sim_score,
        color_score=color_score,
        size_score=size_score,
        texture_score=texture_score,
        num_true_features=true_features,
        is_true=is_true,
        details=details,
    )


def _load_model_for_auto(
    model_name: str,
    class_to_idx: Dict[str, int],
    checkpoint_dir: str,
    device: torch.device,
) -> Optional[Tuple[nn.Module, Dict[str, float], Dict[int, str]]]:
    """Try to load a single model + its metric for auto mode."""
    ckpt_path = os.path.join(
        checkpoint_dir, f"{model_name}_epillid_best.pt"
    )
    if not os.path.exists(ckpt_path):
        return None

    model, inv_class_to_idx = _get_or_load_model(
        model_name=model_name,
        class_to_idx=class_to_idx,
        checkpoint_path=ckpt_path,
        device=device,
    )

    metrics_path = os.path.join(
        checkpoint_dir, f"{model_name}_epillid_best.metrics.json"
    )
    metrics: Dict[str, float] = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            metrics = {}

    return model, metrics, inv_class_to_idx


def compare_pill_images_auto(
    checkpoint_dir: str,
    class_to_idx: Dict[str, int],
    sample_image_path: str,
    query_image_path: str,
    device_str: str | None = None,
    expected_class_name: Optional[str] = None,
    metadata_index: Optional[MedicineMetadataIndex] = None,
    similarity_threshold: float = 0.7,
    color_threshold: float = 0.6,
    size_threshold: float = 0.6,
    texture_threshold: float = 0.6,
    min_true_features: int = 3,
) -> ComparisonResult:
    """Ensemble của resnet50, efficientnet_b0, vit_b_16 (tự sử dụng mô hình nào có sẵn).

    Ý tưởng:
    - Mỗi mô hình đưa ra một ComparisonResult.
    - Trọng số của mỗi mô hình ~ best_val_acc (nếu có), ngược lại = 1.0.
    - Gộp:
        + predicted_class: weighted majority vote.
        + similarity/color/size/texture: trung bình có trọng số.
        + is_true: dựa trên số đặc trưng đạt ngưỡng sau khi gộp.
    """
    device = _resolve_device(device_str)

    available_models: List[Tuple[str, nn.Module, Dict[str, float], Dict[int, str]]] = []
    for name in ["resnet50", "efficientnet_b0", "vit_b_16"]:
        loaded = _load_model_for_auto(name, class_to_idx, checkpoint_dir, device)
        if loaded is not None:
            model, metrics, inv_class_to_idx = loaded
            available_models.append((name, model, metrics, inv_class_to_idx))

    if not available_models:
        raise RuntimeError(
            "Không tìm thấy bất kỳ mô hình nào trong chế độ auto. "
            "Vui lòng train ít nhất một trong các mô hình: resnet50, efficientnet_b0, vit_b_16."
        )

    # Chuẩn bị ảnh & feature chung cho tất cả mô hình
    transform = _get_eval_transform()
    sample_img_pil = focus_on_object(pil_loader(sample_image_path), scale=0.85)
    query_img_pil = focus_on_object(pil_loader(query_image_path), scale=0.85)
    sample_tensor = transform(sample_img_pil).unsqueeze(0)
    query_tensor = transform(query_img_pil).unsqueeze(0)

    # Color/size/texture chỉ phụ thuộc ảnh, tính 1 lần cho tất cả mô hình
    color_score_shared = _compare_colors(sample_img_pil, query_img_pil)
    size_score_shared = _compare_size(sample_img_pil, query_img_pil)
    texture_score_shared = _compare_texture(sample_img_pil, query_img_pil)

    # Thu thập kết quả từng mô hình
    per_model_results: List[ComparisonResult] = []
    class_votes: Dict[str, float] = {}
    for name, model, metrics, inv_class_to_idx in available_models:
        model.eval()
        weight = float(metrics.get("best_val_acc", 1.0))

        query_logits, feat_query = _forward_logits_and_features(model, query_tensor, device)
        pred_idx = int(torch.argmax(query_logits, dim=1).item())
        predicted_class = inv_class_to_idx.get(pred_idx, f"class_{pred_idx}")

        feat_sample = _extract_features(model, sample_tensor, device)
        sim_score = _cosine_similarity(feat_sample, feat_query)

        true_features = _count_true_features(
            sim_score, color_score_shared, size_score_shared, texture_score_shared,
            similarity_threshold, color_threshold, size_threshold, texture_threshold,
        )
        is_true = true_features >= min_true_features

        result = ComparisonResult(
            predicted_class=predicted_class,
            similarity_score=sim_score,
            color_score=color_score_shared,
            size_score=size_score_shared,
            texture_score=texture_score_shared,
            num_true_features=true_features,
            is_true=is_true,
            details={
                "similarity_score": sim_score,
                "color_score": color_score_shared,
                "size_score": size_score_shared,
                "texture_score": texture_score_shared,
                "model_name": name,
                "weight": weight,
            },
        )
        per_model_results.append(result)

        class_votes[predicted_class] = class_votes.get(predicted_class, 0.0) + weight

    # Gộp kết quả
    total_weight = sum(float(r.details.get("weight", 1.0)) for r in per_model_results)
    if total_weight <= 0:
        total_weight = float(len(per_model_results))

    def _weighted_avg(field: str) -> float:
        s = 0.0
        for r in per_model_results:
            w = float(r.details.get("weight", 1.0))
            s += getattr(r, field) * w
        return s / total_weight

    agg_similarity = _weighted_avg("similarity_score")
    # Color/size/texture không phụ thuộc model nên giữ nguyên giá trị chung
    agg_color = color_score_shared
    agg_size = size_score_shared
    agg_texture = texture_score_shared

    agg_true = _count_true_features(
        agg_similarity, agg_color, agg_size, agg_texture,
        similarity_threshold, color_threshold, size_threshold, texture_threshold,
    )

    # Lớp dự đoán cuối cùng: majority vote có trọng số
    final_class = max(class_votes.items(), key=lambda x: x[1])[0]

    class_match = True
    if expected_class_name:
        class_match = final_class == expected_class_name

    metadata_expected = metadata_index.best_match(expected_class_name) if (metadata_index and expected_class_name) else None
    metadata_pred = metadata_index.best_match(final_class) if metadata_index else None
    semantic_group_match = True
    if metadata_expected and metadata_pred:
        if metadata_expected.active_group and metadata_pred.active_group:
            semantic_group_match = metadata_expected.active_group == metadata_pred.active_group

    semantic_penalty = 1 if not semantic_group_match else 0
    effective_true_features = max(0, agg_true - semantic_penalty)
    agg_is_true = class_match and (effective_true_features >= min_true_features)

    details = {
        "similarity_score": agg_similarity,
        "color_score": agg_color,
        "size_score": agg_size,
        "texture_score": agg_texture,
        "num_models": len(per_model_results),
        "class_match": float(1 if class_match else 0),
        "semantic_group_match": float(1 if semantic_group_match else 0),
        "effective_true_features": float(effective_true_features),
    }
    if metadata_expected is not None:
        details["expected_active_group"] = metadata_expected.active_group
        details["expected_disease_vi"] = metadata_expected.disease_vi
        details["expected_medicine_name"] = metadata_expected.medicine_name
    if metadata_pred is not None:
        details["pred_active_group"] = metadata_pred.active_group
        details["pred_disease_vi"] = metadata_pred.disease_vi
        details["pred_medicine_name"] = metadata_pred.medicine_name

    return ComparisonResult(
        predicted_class=final_class,
        similarity_score=agg_similarity,
        color_score=agg_color,
        size_score=agg_size,
        texture_score=agg_texture,
        num_true_features=agg_true,
        is_true=agg_is_true,
        details=details,
    )
