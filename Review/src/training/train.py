from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from ..data.features import PillImageDataset, build_transforms
from ..models.model_factory import create_model, load_checkpoint_class_to_idx


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huấn luyện các mô hình phân loại thuốc")
    parser.add_argument("--data-dir", type=str, default="data_aligned", help="Root data directory")
    parser.add_argument("--model", type=str, default="resnet50",
                        choices=["resnet50", "efficientnet_b0", "vit_b_16"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    # Windows thường gặp lỗi multiprocessing khi num_workers > 0 với một số IDE, nên mặc định set về 0.
    default_workers = 0 if os.name == "nt" else 2
    parser.add_argument("--num-workers", type=int, default=default_workers)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--save-curves", action="store_true", default=True)
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Kỹ thuật làm mượt nhãn (Label Smoothing) giúp mô hình bớt tự tin thái quá, giảm overfitting.",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.2,
        help="Hệ số Alpha cho Mixup. Đặt 0 để tắt. Mixup trộn 2 ảnh lại để tạo dữ liệu mới.",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sử dụng trọng số đã huấn luyện trước (pretrained) từ ImageNet nếu có.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Giới hạn độ lớn của gradient (Gradient Clipping) để tránh bùng nổ gradient khi huấn luyện (<=0 để tắt).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible training curves",
    )
    parser.add_argument(
        "--max-train-val-gap",
        type=float,
        default=0.18,
        help="Cơ chế dừng sớm: nếu độ chính xác tập train cao hơn tập val quá nhiều (overfitting), sẽ dừng lại.",
    )
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=0,
        help="Đóng băng (không huấn luyện) phần backbone trong N epoch đầu để ổn định quá trình học trên tập dữ liệu nhỏ.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.15,
        help="Nếu tập val quá nhỏ, sẽ tự động trích xuất thêm một phần từ tập train để làm tập val (theo tỷ lệ giữ nguyên phân bố lớp).",
    )
    parser.add_argument(
        "--min-val-samples",
        type=int,
        default=24,
        help="Minimum validation sample target before enabling train->val stratified holdout",
    )
    parser.add_argument(
        "--backbone-lr-scale",
        type=float,
        default=0.2,
        help="Hệ số nhân learning rate cho backbone (thường nhỏ hơn so với phần đầu phân loại để tránh quên kiến thức cũ).",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.997,
        help="EMA decay for model parameters (<=0 to disable)",
    )
    parser.add_argument(
        "--tta-views",
        type=int,
        default=3,
        help="Number of TTA views used for validation evaluation",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Enable deterministic CUDA behavior (slower but reproducible).",
    )
    parser.add_argument(
        "--train-metric-every",
        type=int,
        default=3,
        help="Compute exact train metric every N epochs; other epochs use fast approximation.",
    )
    parser.add_argument(
        "--quick-val-tta-views",
        type=int,
        default=1,
        help="Use lightweight TTA views for per-epoch validation; full TTA can run only on save candidates.",
    )
    parser.add_argument(
        "--full-tta-on-save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When quick validation improves, re-evaluate candidate with full TTA before checkpointing.",
    )
    parser.add_argument(
        "--augment-profile",
        type=str,
        default="default",
        choices=["default", "lab6_stable"],
        help="Augmentation profile for train split. 'lab6_stable' adapts Lab6-style recipe.",
    )
    parser.add_argument(
        "--vector-grayscale-prob",
        type=float,
        default=0.0,
        help="Probability to apply grayscale augmentation (Lab6-inspired shape-focused training).",
    )
    return parser.parse_args()


def _set_backbone_trainable(model: nn.Module, model_name: str, trainable: bool) -> None:
    if model_name == "resnet50":
        backbone = [
            model.conv1,
            model.bn1,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        ]
    elif model_name == "efficientnet_b0":
        backbone = [model.features]
    elif model_name == "vit_b_16":
        backbone = [model.encoder]
    else:
        backbone = []

    for module in backbone:
        for p in module.parameters():
            p.requires_grad = trainable


def create_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = False,
    seed: int = 42,
    validation_split: float = 0.15,
    min_val_samples: int = 24,
    augment_profile: str = "default",
    vector_grayscale_prob: float = 0.0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Tạo 3 bộ nạp dữ liệu (DataLoader): train (có biến đổi ảnh), val (cố định), và train-metric (train không biến đổi để đo chỉ số).
    train_root = os.path.join(data_dir, "train")
    val_root = os.path.join(data_dir, "val")

    train_ds = PillImageDataset(
        train_root,
        transform=build_transforms(
            train=True,
            profile=str(augment_profile),
            grayscale_prob=float(vector_grayscale_prob),
        ),
    )
    train_eval_ds = PillImageDataset(
        train_root,
        transform=build_transforms(train=False),
        class_to_idx=train_ds.class_to_idx,
    )
    val_ds = PillImageDataset(
        val_root,
        transform=build_transforms(train=False),
        class_to_idx=train_ds.class_to_idx,
    )

    effective_batch_size = max(1, int(batch_size))
    if len(train_ds) < 8:
        # Với tập dữ liệu quá nhỏ, tự động giảm batch size để phù hợp.
        suggested = max(2, min(8, len(train_ds) // 2))
        effective_batch_size = min(effective_batch_size, suggested)
    effective_batch_size = min(effective_batch_size, max(1, len(train_ds)))

    train_dataset_for_loader = train_ds
    train_eval_dataset_for_loader = train_eval_ds
    val_dataset_for_loader = val_ds

    # If validation is tiny, append a stratified holdout from train for stabler model selection.
    if len(val_ds) < max(1, int(min_val_samples)) and len(train_ds) > len(train_ds.class_to_idx):
        split = max(0.05, min(float(validation_split), 0.4))
        rng = np.random.default_rng(int(seed))

        label_to_indices: Dict[int, List[int]] = {}
        for idx, sample in enumerate(train_ds.samples):
            label_to_indices.setdefault(int(sample.label), []).append(idx)

        holdout_indices: List[int] = []
        keep_indices: List[int] = []
        for _label, indices in label_to_indices.items():
            cls_indices = list(indices)
            rng.shuffle(cls_indices)

            if len(cls_indices) <= 2:
                n_holdout = 0
            else:
                n_holdout = int(round(len(cls_indices) * split))
                n_holdout = max(1, n_holdout)
                n_holdout = min(n_holdout, len(cls_indices) - 1)

            holdout_indices.extend(cls_indices[:n_holdout])
            keep_indices.extend(cls_indices[n_holdout:])

        # Use holdout only if it is meaningful and still leaves train samples.
        if holdout_indices and keep_indices:
            holdout_ds = Subset(train_eval_ds, holdout_indices)
            train_dataset_for_loader = Subset(train_ds, keep_indices)
            train_eval_dataset_for_loader = Subset(train_eval_ds, keep_indices)
            if len(val_ds) > 0:
                val_dataset_for_loader = ConcatDataset([val_ds, holdout_ds])
            else:
                val_dataset_for_loader = holdout_ds

    # Use uniform sampling (disabled class weighting for better train/val gap visibility)
    sampler = None
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(int(seed))

    train_loader = DataLoader(
        train_dataset_for_loader,
        batch_size=effective_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=shuffle_generator,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )
    train_metric_loader = DataLoader(
        train_eval_dataset_for_loader,
        batch_size=min(effective_batch_size, max(1, len(train_eval_dataset_for_loader))),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )
    val_loader = DataLoader(
        val_dataset_for_loader,
        batch_size=min(effective_batch_size, max(len(val_dataset_for_loader), 1)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )
    return train_loader, val_loader, train_metric_loader


def _mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    # Nếu alpha <= 0 thì không dùng Mixup.
    if alpha <= 0:
        return images, labels, labels, 1.0

    # Lấy ngẫu nhiên hệ số lambda từ phân phối Beta.
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=device)
    mixed_images = lam * images + (1.0 - lam) * images[index]
    labels_a = labels
    labels_b = labels[index]
    return mixed_images, labels_a, labels_b, lam

def _tta_logits(model: nn.Module, images: torch.Tensor, views: int = 1) -> torch.Tensor:
    # Nếu số lần nhìn (views) <= 1 thì chỉ dự đoán 1 lần bình thường.
    if views <= 1:
        return model(images)

    # Thêm các phiên bản ảnh lật/xoay để dự đoán (Test Time Augmentation).

    logits_list: List[torch.Tensor] = [model(images)]
    if views >= 2:
        logits_list.append(model(torch.flip(images, dims=[3])))
    if views >= 3:
        logits_list.append(model(torch.flip(images, dims=[2])))
    if views >= 4:
        logits_list.append(model(torch.rot90(images, 1, dims=[2, 3])))
    if views >= 5:
        logits_list.append(model(torch.rot90(images, 3, dims=[2, 3])))

    return torch.stack(logits_list, dim=0).mean(dim=0)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    tta_views: int = 1,
    progress_desc: Optional[str] = None,
) -> Tuple[float, float]:
    # Hàm đánh giá chung cho cả tập train (metric) và tập val.
    model.eval()
    correct, total = 0, 0
    running_loss = 0.0
    use_amp = device.type == "cuda"
    iterator = loader
    if progress_desc:
        iterator = tqdm(loader, desc=progress_desc, leave=False)

    with torch.no_grad():
        for images, labels, _ in iterator:
            images = images.to(device, non_blocking=(device.type == "cuda"))
            labels = labels.to(device, non_blocking=(device.type == "cuda"))
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = _tta_logits(model, images, views=max(1, int(tta_views)))
                loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item() * images.size(0)
    avg_loss = running_loss / max(total, 1)
    return correct / max(total, 1), avg_loss


class ExponentialMovingAverage:
    # Theo dõi phiên bản trung bình (smoothed) của tham số mô hình để giảm dao động khi đánh giá validation.
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    def update(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if not p.requires_grad or name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=(1.0 - self.decay))

    def apply(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        backup: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if not p.requires_grad or name not in self.shadow:
                continue
            backup[name] = p.detach().clone()
            p.data.copy_(self.shadow[name].data)
        return backup

    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        for name, p in model.named_parameters():
            if name in backup:
                p.data.copy_(backup[name].data)


def _split_parameter_groups(
    model: nn.Module,
    model_name: str,
    base_lr: float,
    backbone_lr_scale: float,
) -> List[Dict[str, object]]:
    head_prefixes = {
        "resnet50": ("fc.",),
        "efficientnet_b0": ("classifier.",),
        "vit_b_16": ("heads.",),
    }.get(model_name, tuple())

    head_params: List[torch.nn.Parameter] = []
    backbone_params: List[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if head_prefixes and name.startswith(head_prefixes):
            head_params.append(p)
        else:
            backbone_params.append(p)

    if not head_params or not backbone_params:
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": float(base_lr)}]

    scaled_backbone_lr = float(base_lr) * max(0.05, min(float(backbone_lr_scale), 1.0))
    return [
        {"params": backbone_params, "lr": scaled_backbone_lr},
        {"params": head_params, "lr": float(base_lr)},
    ]


def _dataset_len(dataset: object) -> int:
    try:
        return int(len(dataset))
    except Exception:
        return 0


def _plot_training_curves(history: Dict[str, List[float]], output_path: str) -> None:
    import matplotlib.pyplot as plt

    epochs = history.get("epoch", [])
    if not epochs:
        return

    def _ema(values: List[float], alpha: float = 0.35) -> List[float]:
        if not values:
            return []
        out = [float(values[0])]
        for v in values[1:]:
            out.append(alpha * float(v) + (1.0 - alpha) * out[-1])
        return out

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])

    train_loss_s = _ema(train_loss)
    val_loss_s = _ema(val_loss)
    train_acc_s = _ema(train_acc)
    val_acc_s = _ema(val_acc)

    if not val_loss_s or not val_acc_s:
        return

    best_idx = int(np.argmin(np.array(val_loss_s, dtype=np.float32)))
    best_epoch = int(epochs[best_idx])

    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Training Curves (Smoothed) - Stable Train/Val", fontsize=18, fontweight="bold")

    ax = axes[0]
    ax.plot(epochs, train_loss, color="#1f77b4", alpha=0.25, linewidth=1.0)
    ax.plot(epochs, val_loss, color="#ff7f0e", alpha=0.25, linewidth=1.0)
    ax.plot(epochs, train_loss_s, color="#1f77b4", label="Train Loss", linewidth=2.2)
    ax.plot(epochs, val_loss_s, color="#ff7f0e", linestyle="--", label="Val Loss", linewidth=2.2)
    ax.fill_between(epochs, train_loss_s, val_loss_s, color="gray", alpha=0.10, label="Gap")
    ax.axvline(best_epoch, color="green", linestyle=":", linewidth=1.3, label=f"Best epoch ({best_epoch})")
    ax.set_title("Loss vs. Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CrossEntropy Loss")
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(epochs, train_acc, color="#1f77b4", alpha=0.25, linewidth=1.0)
    ax.plot(epochs, val_acc, color="#ff7f0e", alpha=0.25, linewidth=1.0)
    ax.plot(epochs, train_acc_s, color="#1f77b4", label="Train Acc", linewidth=2.2)
    ax.plot(epochs, val_acc_s, color="#ff7f0e", linestyle="--", label="Val Acc", linewidth=2.2)
    ax.fill_between(epochs, train_acc_s, val_acc_s, color="gray", alpha=0.10, label="Gap")
    ax.axvline(best_epoch, color="green", linestyle=":", linewidth=1.3, label=f"Best epoch ({best_epoch})")
    ax.annotate(
        f"Val Acc = {val_acc_s[-1]:.4f}",
        xy=(epochs[-1], val_acc_s[-1]),
        xytext=(epochs[-1], max(0.0, val_acc_s[-1] - 0.06)),
        color="green",
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="green", lw=1.0),
    )
    ax.set_title("Accuracy vs. Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _grad_global_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        total += float(torch.sum(g * g).item())
    return float(total ** 0.5)


def _is_diag_epoch(epoch: int) -> bool:
    return 4 <= epoch <= 7


def _print_terminal_review_header(max_epochs: int, patience: int, freeze_backbone_epochs: int, base_lr: float) -> None:
    print(f"[START] Training review | Max {max_epochs} epoch | Patience={patience}")
    if freeze_backbone_epochs > 0:
        print(
            f"    Stage 1 (frozen): epoch 1-{freeze_backbone_epochs}, LR={base_lr:g}"
        )
        print(
            f"    Stage 2 (unfreeze): epoch {freeze_backbone_epochs + 1}+, LR={base_lr:g}"
        )
    else:
        print(f"    Stage 1 (train-all): epoch 1+, LR={base_lr:g}")
    print("Epoch |  TrLoss |  VaLoss |  TrAcc |  VaAcc | Stage   | Status")
    print("-" * 76)


def _stage_name(epoch: int, freeze_backbone_epochs: int) -> str:
    if freeze_backbone_epochs > 0 and epoch <= freeze_backbone_epochs:
        return "stage-1"
    if freeze_backbone_epochs > 0:
        return "stage-2"
    return "stage-1"


def _resolve_class_to_idx(dataset: torch.utils.data.Dataset) -> Dict[str, int]:
    # train_loader.dataset can be a Subset/ConcatDataset after stratified holdout.
    mapping = getattr(dataset, "class_to_idx", None)
    if isinstance(mapping, dict) and mapping:
        return mapping

    if isinstance(dataset, Subset):
        return _resolve_class_to_idx(dataset.dataset)

    if isinstance(dataset, ConcatDataset):
        for child in dataset.datasets:
            child_mapping = getattr(child, "class_to_idx", None)
            if isinstance(child_mapping, dict) and child_mapping:
                return child_mapping
            if isinstance(child, (Subset, ConcatDataset)):
                nested_mapping = _resolve_class_to_idx(child)
                if nested_mapping:
                    return nested_mapping

    raise AttributeError("Unable to resolve class_to_idx from dataset")


def train(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    # Full deterministic seed setup for reproducible curves.
    seed = int(getattr(args, "seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    deterministic = bool(getattr(args, "deterministic", False))
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = (not deterministic)

    train_metric_every = max(1, int(getattr(args, "train_metric_every", 3)))
    total_tta_views = max(1, int(getattr(args, "tta_views", 1)))
    quick_val_tta_views = max(
        1,
        min(
            int(getattr(args, "quick_val_tta_views", 1)),
            total_tta_views,
        ),
    )
    full_tta_on_save = bool(getattr(args, "full_tta_on_save", True))

    # Graceful device fallback keeps CLI behavior predictable.
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA không khả dụng, chuyển sang CPU.")
        args.device = "cpu"

    device = torch.device(args.device)
    use_amp = device.type == "cuda"
    pin_memory = device.type == "cuda"

    train_loader, val_loader, train_metric_loader = create_dataloaders(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        pin_memory=pin_memory,
        seed=int(getattr(args, "seed", 42)),
        validation_split=float(getattr(args, "validation_split", 0.15)),
        min_val_samples=int(getattr(args, "min_val_samples", 24)),
        augment_profile=str(getattr(args, "augment_profile", "default")),
        vector_grayscale_prob=float(getattr(args, "vector_grayscale_prob", 0.0)),
    )

    class_to_idx = _resolve_class_to_idx(train_loader.dataset)
    num_classes = len(class_to_idx)
    # Use uniform class weights to keep train/val gap diagnostics easier to interpret.
    class_weight_tensor = None

    model, _ = create_model(
        args.model,
        num_classes=num_classes,
        pretrained=bool(getattr(args, "pretrained", True)),
        fallback_to_random=True,
    )
    model = model.to(device)

    label_smoothing = float(getattr(args, "label_smoothing", 0.1))
    mixup_alpha = float(getattr(args, "mixup_alpha", 0.2))
    n_train = _dataset_len(train_loader.dataset)
    n_val = _dataset_len(val_loader.dataset)

    # Gap guard is tuned differently for tiny datasets where metrics are noisier.
    max_large_gap_epochs = 2
    gap_guard_start_epoch = 3
    if n_train <= 64:
        # On very small datasets, heavy regularization often pushes val acc to 0 for CNN backbones.
        label_smoothing = min(label_smoothing, 0.02)
        mixup_alpha = min(mixup_alpha, 0.04)
        if args.model in {"resnet50", "efficientnet_b0"} and float(args.lr) < 1.8e-4:
            args.lr = 1.8e-4
        if int(getattr(args, "freeze_backbone_epochs", 0)) > 2:
            args.freeze_backbone_epochs = 2
        if int(getattr(args, "early_stop_patience", 5)) < 8:
            args.early_stop_patience = 8
        max_large_gap_epochs = 1
        gap_guard_start_epoch = 5

    print(
        "[INFO] data_profile "
        f"train_samples={n_train} val_samples={n_val} "
        f"train_batches={len(train_loader)} val_batches={len(val_loader)} "
        f"label_smoothing={label_smoothing:.3f} mixup_alpha={mixup_alpha:.3f} "
        f"lr={float(args.lr):.6f} freeze_backbone_epochs={int(getattr(args, 'freeze_backbone_epochs', 0))} "
        f"augment_profile={str(getattr(args, 'augment_profile', 'default'))} "
        f"vector_grayscale_prob={float(getattr(args, 'vector_grayscale_prob', 0.0)):.2f}"
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weight_tensor,
        label_smoothing=label_smoothing,
    )
    backbone_lr_scale = float(getattr(args, "backbone_lr_scale", 0.2))
    param_groups = _split_parameter_groups(
        model=model,
        model_name=args.model,
        base_lr=float(args.lr),
        backbone_lr_scale=backbone_lr_scale,
    )
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    for pg in optimizer.param_groups:
        pg["initial_lr"] = float(pg["lr"])
    base_lr = float(args.lr)
    warmup_epochs = 3
    # Warmup 3 epoch + cosine decay giúp head ổn định trước khi backbone bắt đầu unfreeze.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(2, int(args.epochs) - warmup_epochs),
        eta_min=base_lr * 0.05,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    freeze_backbone_epochs = int(getattr(args, "freeze_backbone_epochs", 0))
    ema_decay = float(getattr(args, "ema_decay", 0.997))
    ema: Optional[ExponentialMovingAverage] = None
    if ema_decay > 0:
        ema = ExponentialMovingAverage(model, decay=min(max(ema_decay, 0.9), 0.9999))

    metrics_path = os.path.join(args.output_dir, f"{args.model}_epillid_best.metrics.json")
    ckpt_path = os.path.join(args.output_dir, f"{args.model}_epillid_best.pt")
    existing_metrics: Dict[str, float] = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                existing_metrics = obj
        except Exception as e:
            LOGGER.warning("Failed to load existing metrics from %s: %s", metrics_path, e)
            existing_metrics = {}

    # Nếu thiếu checkpoint nhưng còn metrics cũ, reset để buộc save lại checkpoint mới.
    if not os.path.exists(ckpt_path) and existing_metrics:
        LOGGER.warning(
            "Checkpoint file missing for %s while metrics exists. Resetting stale metrics.",
            args.model,
        )
        existing_metrics = {}

    # Nếu checkpoint cũ dùng taxonomy class khác dataset hiện tại, reset best để bắt buộc ghi đè.
    if os.path.exists(ckpt_path):
        try:
            ckpt_class_to_idx = load_checkpoint_class_to_idx(ckpt_path, map_location="cpu")
            current_class_to_idx = class_to_idx
            if ckpt_class_to_idx and ckpt_class_to_idx != current_class_to_idx:
                LOGGER.warning(
                    "Checkpoint class mapping mismatch for %s. Resetting best metrics to retrain on new label space.",
                    args.model,
                )
                existing_metrics = {}
        except Exception as e:
            LOGGER.warning("Failed to inspect checkpoint class mapping for %s: %s", args.model, e)

    best_acc = float(existing_metrics.get("best_val_acc", -1.0))
    best_epoch = int(existing_metrics.get("epochs", 0))
    metrics: Dict[str, float] = dict(existing_metrics)
    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    epochs_without_improve = 0
    epochs_with_large_gap = 0
    last_full_train_acc: Optional[float] = None

    _print_terminal_review_header(
        max_epochs=int(args.epochs),
        patience=int(args.early_stop_patience),
        freeze_backbone_epochs=freeze_backbone_epochs,
        base_lr=base_lr,
    )

    for epoch in range(1, args.epochs + 1):
        if freeze_backbone_epochs > 0:
            if epoch <= freeze_backbone_epochs:
                _set_backbone_trainable(model, args.model, trainable=False)
            elif epoch == freeze_backbone_epochs + 1:
                _set_backbone_trainable(model, args.model, trainable=True)

        if epoch <= warmup_epochs:
            warmup_scale = 0.25 + 0.75 * (epoch / warmup_epochs)
            for pg in optimizer.param_groups:
                base_group_lr = pg.get("initial_lr")
                if not isinstance(base_group_lr, (int, float)):
                    base_group_lr = pg.get("lr", base_lr)
                pg["lr"] = float(base_group_lr) * warmup_scale

        model.train()
        running_loss = 0.0
        running_correct = 0.0
        running_total = 0
        pbar = tqdm(train_loader, desc=f"Vong lap {epoch}/{args.epochs}")
        epoch_grad_norm_sum = 0.0
        epoch_grad_norm_max = 0.0
        epoch_batches = 0
        for images, labels, _ in pbar:
            images = images.to(device, non_blocking=(device.type == "cuda"))
            labels = labels.to(device, non_blocking=(device.type == "cuda"))
            mixed_images, labels_a, labels_b, lam = _mixup_batch(
                images, labels, mixup_alpha, device
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(mixed_images)
                loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(
                    logits, labels_b
                )
            scaler.scale(loss).backward()

            if float(getattr(args, "grad_clip_norm", 0.0)) > 0:
                scaler.unscale_(optimizer)

            batch_grad_norm = _grad_global_norm(model)
            epoch_grad_norm_sum += batch_grad_norm
            epoch_grad_norm_max = max(epoch_grad_norm_max, batch_grad_norm)
            epoch_batches += 1

            if float(getattr(args, "grad_clip_norm", 0.0)) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))

            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_a = (preds == labels_a).sum().item()
            correct_b = (preds == labels_b).sum().item()
            running_correct += (lam * correct_a) + ((1.0 - lam) * correct_b)
            running_total += labels.size(0)
            pbar.set_postfix(mat_mat=loss.item())

        epoch_loss = running_loss / max(n_train, 1)
        # Use fast approximate train metric most epochs; run exact clean-train evaluation periodically.
        approx_train_acc = float(running_correct / max(running_total, 1))
        compute_full_train_metric = (
            epoch == 1
            or epoch == int(args.epochs)
            or (epoch % train_metric_every == 0)
        )

        ema_backup: Optional[Dict[str, torch.Tensor]] = None
        if ema is not None:
            ema_backup = ema.apply(model)

        if compute_full_train_metric:
            train_eval_start = time.perf_counter()
            print(f"[INFO] Eval train metric (epoch {epoch}, mode=full)")
            train_acc, _train_eval_loss = evaluate(
                model,
                train_metric_loader,
                device,
                criterion,
                tta_views=1,
                progress_desc=f"Eval train e{epoch}",
            )
            print(f"[INFO] Eval train done in {time.perf_counter() - train_eval_start:.1f}s")
            train_metric_mode = "full"
            last_full_train_acc = float(train_acc)
        else:
            train_acc = approx_train_acc
            train_metric_mode = "approx"

        val_quick_start = time.perf_counter()
        print(
            f"[INFO] Eval val quick (epoch {epoch}, tta_views={quick_val_tta_views})"
        )
        val_acc_quick, val_loss_quick = evaluate(
            model,
            val_loader,
            device,
            criterion,
            tta_views=quick_val_tta_views,
            progress_desc=f"Eval val quick e{epoch}",
        )
        print(f"[INFO] Eval val quick done in {time.perf_counter() - val_quick_start:.1f}s")
        val_acc = float(val_acc_quick)
        val_loss = float(val_loss_quick)
        val_metric_mode = f"quick@{quick_val_tta_views}"

        candidate_improved = val_acc_quick > (best_acc + 1e-4)
        if (
            full_tta_on_save
            and total_tta_views > quick_val_tta_views
            and candidate_improved
            and best_acc >= 0.0
        ):
            val_full_start = time.perf_counter()
            print(
                f"[INFO] Eval val full (epoch {epoch}, tta_views={total_tta_views})"
            )
            val_acc, val_loss = evaluate(
                model,
                val_loader,
                device,
                criterion,
                tta_views=total_tta_views,
                progress_desc=f"Eval val full e{epoch}",
            )
            print(f"[INFO] Eval val full done in {time.perf_counter() - val_full_start:.1f}s")
            val_metric_mode = f"full@{total_tta_views}"

        if ema is not None and ema_backup is not None:
            ema.restore(model, ema_backup)

        history["epoch"].append(epoch)
        history["train_loss"].append(float(epoch_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))

        gap = float(train_acc - val_acc)
        if compute_full_train_metric and gap > float(getattr(args, "max_train_val_gap", 0.18)) and epoch >= gap_guard_start_epoch:
            epochs_with_large_gap += 1
        else:
            epochs_with_large_gap = 0

        prev_lr = float(max(pg["lr"] for pg in optimizer.param_groups))
        if epoch > warmup_epochs:
            scheduler.step()
        next_lr = float(optimizer.param_groups[0]["lr"])

        if _is_diag_epoch(epoch):
            avg_grad_norm = epoch_grad_norm_sum / max(epoch_batches, 1)
            print(
                "[DIAG] "
                f"epoch={epoch} lr_before={prev_lr:.8f} lr_after={next_lr:.8f} "
                f"grad_norm_avg={avg_grad_norm:.6f} grad_norm_max={epoch_grad_norm_max:.6f} "
                f"batches={epoch_batches} train_samples={running_total} val_samples={n_val}"
            )
            if next_lr != prev_lr:
                print(
                    "[DIAG] "
                    f"scheduler_event=CosineAnnealingLR lr_changed_from={prev_lr:.8f} to={next_lr:.8f}"
                )

        saved_this_epoch = False
        # Save only meaningful improvements to avoid churn from tiny metric fluctuations.
        if val_acc > best_acc + 1e-4:
            best_acc = val_acc
            best_epoch = epoch
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "class_to_idx": class_to_idx,
                    "val_acc": val_acc,
                },
                ckpt_path,
            )
            metrics = {
                "best_val_acc": float(best_acc),
                "epochs": best_epoch,
                "model": args.model,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "label_smoothing": label_smoothing,
                "mixup_alpha": mixup_alpha,
                "backbone_lr_scale": backbone_lr_scale,
                "ema_decay": ema_decay,
                "tta_views": int(getattr(args, "tta_views", 1)),
            }
            epochs_without_improve = 0
            saved_this_epoch = True
        else:
            epochs_without_improve += 1

        stage = _stage_name(epoch, freeze_backbone_epochs)
        if saved_this_epoch:
            status = "[SAVED]"
        else:
            status = f"wait {epochs_without_improve}/{int(args.early_stop_patience)}"

        print(
            f"{epoch:>5} | {epoch_loss:>7.4f} | {val_loss:>7.4f} | "
            f"{train_acc:>6.4f} | {val_acc:>6.4f} | {stage:<7} | {status} "
            f"[{train_metric_mode},{val_metric_mode}]"
        )

        if epochs_without_improve >= args.early_stop_patience:
            print(
                f"Dung som tai vong {epoch} "
                f"(khong cai thien trong {args.early_stop_patience} vong)."
            )
            break

        if epochs_with_large_gap >= max_large_gap_epochs:
            print(
                f"Dung som tai vong {epoch} "
                f"(train-val gap={gap:.4f} vuot nguong {float(getattr(args, 'max_train_val_gap', 0.18)):.4f})."
            )
            break

    if metrics:
        old_best = float(existing_metrics.get("best_val_acc", 0.0))
        metrics["best_val_acc"] = float(max(old_best, float(metrics.get("best_val_acc", 0.0))))
        if metrics["best_val_acc"] <= old_best:
            metrics["epochs"] = int(existing_metrics.get("epochs", metrics.get("epochs", 0)))
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    history_path = os.path.join(args.output_dir, f"{args.model}_epillid_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    if args.save_curves:
        curves_path = os.path.join(args.output_dir, f"{args.model}_training_curves.png")
        _plot_training_curves(history, curves_path)


if __name__ == "__main__":
    train()

