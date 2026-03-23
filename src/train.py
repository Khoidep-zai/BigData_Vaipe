from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .features import PillImageDataset, build_transforms
from .models import create_model, load_checkpoint_class_to_idx


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pill classification models")
    parser.add_argument("--data-dir", type=str, default="data_aligned", help="Root data directory")
    parser.add_argument("--model", type=str, default="resnet50",
                        choices=["resnet50", "efficientnet_b0", "vit_b_16"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    # Windows thường dễ lỗi multiprocessing khi num_workers > 0 trong một số môi trường IDE.
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
        help="Label smoothing for CrossEntropyLoss",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.2,
        help="Mixup alpha. Set 0 to disable mixup",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pretrained weights when available",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Clip gradient norm (<=0 to disable)",
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
        help="Early-stop guard: if train_acc - val_acc exceeds this repeatedly, stop to avoid curve divergence",
    )
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=0,
        help="Freeze backbone for first N epochs to keep train/val curves more stable on very small datasets",
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
    data_dir: str, batch_size: int, num_workers: int, pin_memory: bool = False
) -> Tuple[DataLoader, DataLoader]:
    train_root = os.path.join(data_dir, "train")
    val_root = os.path.join(data_dir, "val")

    train_ds = PillImageDataset(train_root, transform=build_transforms(train=True))
    val_ds = PillImageDataset(
        val_root,
        transform=build_transforms(train=False),
        class_to_idx=train_ds.class_to_idx,
    )

    # Use uniform sampling (disabled class weighting for better train/val gap visibility)
    sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def _mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return images, labels, labels, 1.0

    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=device)
    mixed_images = lam * images + (1.0 - lam) * images[index]
    labels_a = labels
    labels_b = labels[index]
    return mixed_images, labels_a, labels_b, lam


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    correct, total = 0, 0
    running_loss = 0.0
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item() * images.size(0)
    avg_loss = running_loss / max(total, 1)
    return correct / max(total, 1), avg_loss


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


def train(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    seed = int(getattr(args, "seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Fallback khi chọn cuda nhưng không có GPU
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA không khả dụng, chuyển sang CPU.")
        args.device = "cpu"

    device = torch.device(args.device)
    use_amp = device.type == "cuda"
    pin_memory = device.type == "cuda"

    train_loader, val_loader = create_dataloaders(
        args.data_dir, args.batch_size, args.num_workers, pin_memory=pin_memory
    )

    num_classes = len(train_loader.dataset.class_to_idx)
    # Use uniform loss weights (disabled class weighting for better train/val gap visibility)
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
    criterion = nn.CrossEntropyLoss(
        weight=class_weight_tensor,
        label_smoothing=label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    base_lr = float(args.lr)
    warmup_epochs = 3
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=2, min_lr=base_lr * 0.1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    freeze_backbone_epochs = int(getattr(args, "freeze_backbone_epochs", 0))

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
            current_class_to_idx = train_loader.dataset.class_to_idx
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
            warmup_lr = base_lr * (0.4 + 0.6 * (epoch / warmup_epochs))
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        model.train()
        running_loss = 0.0
        running_correct = 0.0
        running_total = 0
        pbar = tqdm(train_loader, desc=f"Vong lap {epoch}/{args.epochs}")
        epoch_grad_norm_sum = 0.0
        epoch_grad_norm_max = 0.0
        epoch_batches = 0
        for images, labels, _ in pbar:
            images = images.to(device)
            labels = labels.to(device)
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

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_a = (preds == labels_a).sum().item()
            correct_b = (preds == labels_b).sum().item()
            running_correct += (lam * correct_a) + ((1.0 - lam) * correct_b)
            running_total += labels.size(0)
            pbar.set_postfix(mat_mat=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = running_correct / max(running_total, 1)
        val_acc, val_loss = evaluate(model, val_loader, device, criterion)

        history["epoch"].append(epoch)
        history["train_loss"].append(float(epoch_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))

        gap = float(train_acc - val_acc)
        if gap > float(getattr(args, "max_train_val_gap", 0.18)) and epoch >= 3:
            epochs_with_large_gap += 1
        else:
            epochs_with_large_gap = 0

        prev_lr = float(optimizer.param_groups[0]["lr"])
        if epoch > warmup_epochs:
            scheduler.step(val_loss)
        next_lr = float(optimizer.param_groups[0]["lr"])

        if _is_diag_epoch(epoch):
            avg_grad_norm = epoch_grad_norm_sum / max(epoch_batches, 1)
            print(
                "[DIAG] "
                f"epoch={epoch} lr_before={prev_lr:.8f} lr_after={next_lr:.8f} "
                f"grad_norm_avg={avg_grad_norm:.6f} grad_norm_max={epoch_grad_norm_max:.6f} "
                f"batches={epoch_batches} train_samples={running_total} val_samples={len(val_loader.dataset)}"
            )
            if next_lr != prev_lr:
                print(
                    "[DIAG] "
                    f"scheduler_event=ReduceLROnPlateau lr_changed_from={prev_lr:.8f} to={next_lr:.8f}"
                )

        saved_this_epoch = False
        if val_acc > best_acc + 1e-4:
            best_acc = val_acc
            best_epoch = epoch
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "class_to_idx": train_loader.dataset.class_to_idx,
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
            f"{train_acc:>6.4f} | {val_acc:>6.4f} | {stage:<7} | {status}"
        )

        if epochs_without_improve >= args.early_stop_patience:
            print(
                f"Dung som tai vong {epoch} "
                f"(khong cai thien trong {args.early_stop_patience} vong)."
            )
            break

        if epochs_with_large_gap >= 2:
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

