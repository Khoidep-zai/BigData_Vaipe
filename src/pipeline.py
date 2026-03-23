from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from .features import PillImageDataset, build_transforms
from .models import load_checkpoint, load_checkpoint_class_to_idx
from .train import train as train_one_model

DEFAULT_MODELS = ["resnet50", "efficientnet_b0", "vit_b_16"]


@dataclass
class ModelEvalResult:
    model: str
    accuracy: float
    macro_f1: float
    num_samples: int
    checkpoint: str


@dataclass
class PipelineSummary:
    started_at: str
    finished_at: str
    elapsed_seconds: float
    data_dir: str
    test_dir: str
    models_dir: str
    report_dir: str
    trained_models: List[str]
    best_model: str
    ensemble_model_name: str


def _stage(stage_idx: int, total_stages: int, message: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [Giai doan {stage_idx}/{total_stages}] {message}", flush=True)


def _resolve_device(device_str: str) -> str:
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[CANH_BAO] CUDA khong kha dung, tu dong chuyen sang CPU.", flush=True)
        return "cpu"
    return device_str


def _is_valid_dataset_root(root: Path) -> bool:
    return (root / "train").exists() and (root / "val").exists() and (root / "test").exists()


def _count_train_classes(root: Path) -> int:
    train_root = root / "train"
    if not train_root.exists():
        return 0
    return len([d for d in train_root.iterdir() if d.is_dir() and not d.name.startswith(".")])


def discover_data_dir(preferred: Optional[str] = None) -> str:
    if preferred:
        root = Path(preferred)
        if not _is_valid_dataset_root(root):
            raise FileNotFoundError(
                f"Invalid data dir: {root}. Expected train/val/test subfolders."
            )
        return str(root)

    candidates = [Path("data_aligned"), Path("data")]
    valid = [c for c in candidates if _is_valid_dataset_root(c)]
    if not valid:
        raise FileNotFoundError(
            "No valid dataset root found. Please provide --data-dir with train/val/test."
        )

    # Prefer richer class layout when both exist.
    valid.sort(key=_count_train_classes, reverse=True)
    return str(valid[0])


def _build_loader(test_root: Path, batch_size: int, num_workers: int) -> Tuple[PillImageDataset, DataLoader]:
    dataset = PillImageDataset(str(test_root), transform=build_transforms(train=False))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def _safe_load_metrics(metrics_path: Path) -> Dict[str, float]:
    if not metrics_path.exists():
        return {}
    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return {str(k): float(v) for k, v in obj.items() if isinstance(v, (int, float))}
    except Exception:
        pass
    return {}


def _portable_checkpoint_path(checkpoint_path: Path) -> str:
    try:
        rel = checkpoint_path.resolve().relative_to(Path.cwd().resolve())
        return rel.as_posix()
    except Exception:
        return checkpoint_path.as_posix()


def _evaluate_single_model(
    model_name: str,
    checkpoint_path: Path,
    loader: DataLoader,
    dataset_class_to_idx: Dict[str, int],
    device: torch.device,
) -> Tuple[ModelEvalResult, List[str], List[str]]:
    ckpt_class_to_idx = load_checkpoint_class_to_idx(str(checkpoint_path), map_location=device)
    num_classes = len(ckpt_class_to_idx) if ckpt_class_to_idx else len(dataset_class_to_idx)

    model = load_checkpoint(
        model_name=model_name,
        num_classes=num_classes,
        checkpoint_path=str(checkpoint_path),
        map_location=device,
    ).to(device)
    model.eval()

    inv_true = {v: k for k, v in dataset_class_to_idx.items()}
    inv_pred = {v: k for k, v in (ckpt_class_to_idx or dataset_class_to_idx).items()}

    y_true: List[str] = []
    y_pred: List[str] = []

    with torch.no_grad():
        for images, labels, _paths in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            for t_idx, p_idx in zip(labels.tolist(), preds.tolist()):
                y_true.append(inv_true.get(int(t_idx), f"class_{int(t_idx)}"))
                y_pred.append(inv_pred.get(int(p_idx), f"class_{int(p_idx)}"))

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    result = ModelEvalResult(
        model=model_name,
        accuracy=acc,
        macro_f1=macro_f1,
        num_samples=len(y_true),
        checkpoint=_portable_checkpoint_path(checkpoint_path),
    )
    return result, y_true, y_pred


def _evaluate_ensemble(
    model_names: Sequence[str],
    models_dir: Path,
    loader: DataLoader,
    dataset_class_to_idx: Dict[str, int],
    device: torch.device,
) -> Tuple[ModelEvalResult, List[str], List[str]]:
    inv_true = {v: k for k, v in dataset_class_to_idx.items()}

    all_probs: List[Tuple[Dict[int, str], float, torch.Tensor]] = []
    y_true_idx: Optional[torch.Tensor] = None

    for model_name in model_names:
        ckpt_path = models_dir / f"{model_name}_epillid_best.pt"
        if not ckpt_path.exists():
            continue

        ckpt_class_to_idx = load_checkpoint_class_to_idx(str(ckpt_path), map_location=device)
        num_classes = len(ckpt_class_to_idx) if ckpt_class_to_idx else len(dataset_class_to_idx)
        model = load_checkpoint(
            model_name=model_name,
            num_classes=num_classes,
            checkpoint_path=str(ckpt_path),
            map_location=device,
        ).to(device)
        model.eval()

        inv_pred = {v: k for k, v in (ckpt_class_to_idx or dataset_class_to_idx).items()}

        metrics_path = models_dir / f"{model_name}_epillid_best.metrics.json"
        metrics = _safe_load_metrics(metrics_path)
        weight = float(metrics.get("best_val_acc", 1.0))
        if weight <= 0:
            weight = 1.0

        model_probs: List[torch.Tensor] = []
        model_labels: List[torch.Tensor] = []
        with torch.no_grad():
            for images, labels, _paths in loader:
                images = images.to(device)
                probs = F.softmax(model(images), dim=1)
                model_probs.append(probs.cpu())
                model_labels.append(labels.cpu())

        concat_probs = torch.cat(model_probs, dim=0)
        labels_cat = torch.cat(model_labels, dim=0)
        if y_true_idx is None:
            y_true_idx = labels_cat

        all_probs.append((inv_pred, weight, concat_probs))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not all_probs or y_true_idx is None:
        raise RuntimeError("No checkpoints available to build ensemble.")

    y_true: List[str] = []
    y_pred: List[str] = []

    for sample_idx, true_idx in enumerate(y_true_idx.tolist()):
        votes: Dict[str, float] = {}
        for inv_pred, weight, probs in all_probs:
            row = probs[sample_idx]
            for cls_idx, cls_prob in enumerate(row.tolist()):
                cls_name = inv_pred.get(int(cls_idx), f"class_{int(cls_idx)}")
                votes[cls_name] = votes.get(cls_name, 0.0) + weight * float(cls_prob)

        pred_name = max(votes.items(), key=lambda x: x[1])[0]
        y_true.append(inv_true.get(int(true_idx), f"class_{int(true_idx)}"))
        y_pred.append(pred_name)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    result = ModelEvalResult(
        model="ensemble_weighted",
        accuracy=acc,
        macro_f1=macro_f1,
        num_samples=len(y_true),
        checkpoint="multiple",
    )
    return result, y_true, y_pred


def _write_summary_csv(rows: Iterable[ModelEvalResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "accuracy", "macro_f1", "num_samples", "checkpoint"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _plot_score_comparison(rows: List[ModelEvalResult], chart_path: Path) -> None:
    chart_path.parent.mkdir(parents=True, exist_ok=True)

    model_names = [r.model for r in rows]
    accs = [r.accuracy for r in rows]
    f1s = [r.macro_f1 for r in rows]

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(12, 5))

    x = list(range(len(model_names)))
    width = 0.36
    ax.bar([i - width / 2 for i in x], accs, width=width, label="Accuracy")
    ax.bar([i + width / 2 for i in x], f1s, width=width, label="Macro F1")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Evaluation Comparison")
    ax.legend(loc="best")

    for i, val in enumerate(accs):
        ax.text(i - width / 2, val + 0.02, f"{val:.3f}", ha="center", fontsize=9)
    for i, val in enumerate(f1s):
        ax.text(i + width / 2, val + 0.02, f"{val:.3f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)


def _plot_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Keep annotations lightweight if matrix is not too large.
    if len(labels) <= 15:
        threshold = cm.max() / 2.0 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    int(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > threshold else "black",
                    fontsize=8,
                )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: train all models, evaluate, and export visual reports"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Dataset root containing train/val/test. If omitted, auto-discovery is used.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated backbone names",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 2)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument(
        "--report-dir",
        type=str,
        default="models/reports/latest",
        help="Directory for summary CSV/JSON/charts",
    )
    parser.add_argument(
        "--save-curves",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save training curves per model",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pretrained weights if available",
    )
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace | None = None) -> PipelineSummary:
    if args is None:
        args = parse_args()

    total_stages = 5
    started_at_dt = datetime.now()
    started_at = started_at_dt.isoformat(timespec="seconds")

    # Stage 1: discover input/output templates.
    _stage(1, total_stages, "Phat hien du lieu va tao thu muc dau ra")
    data_dir = discover_data_dir(args.data_dir)
    test_dir = str(Path(data_dir) / "test")
    models_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    device_name = _resolve_device(args.device)
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_names:
        raise ValueError("No model names provided.")

    # Stage 2: train all requested models.
    _stage(2, total_stages, f"Train {len(model_names)} mo hinh: {', '.join(model_names)}")
    trained_models: List[str] = []
    for idx, model_name in enumerate(model_names, start=1):
        _stage(2, total_stages, f"[{idx}/{len(model_names)}] Dang train mo hinh: {model_name}")
        train_args = argparse.Namespace(
            data_dir=data_dir,
            model=model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            label_smoothing=float(getattr(args, "label_smoothing", 0.1)),
            mixup_alpha=float(getattr(args, "mixup_alpha", 0.2)),
            num_workers=args.num_workers,
            device=device_name,
            output_dir=str(models_dir),
            early_stop_patience=args.early_stop_patience,
            save_curves=args.save_curves,
            pretrained=args.pretrained,
        )
        train_one_model(train_args)

        ckpt = models_dir / f"{model_name}_epillid_best.pt"
        if ckpt.exists():
            trained_models.append(model_name)
        else:
            print(f"[CANH_BAO] Khong tim thay checkpoint sau train: {ckpt}", flush=True)

    if not trained_models:
        raise RuntimeError("Training finished but no checkpoints were produced.")

    # Stage 3: evaluate each model and weighted ensemble.
    _stage(3, total_stages, "Danh gia toan bo mo hinh da train va ensemble co trong so")
    device = torch.device(device_name)
    dataset, loader = _build_loader(Path(test_dir), batch_size=args.batch_size, num_workers=args.num_workers)

    eval_rows: List[ModelEvalResult] = []
    pred_cache: Dict[str, Tuple[List[str], List[str]]] = {}

    for model_name in trained_models:
        ckpt = models_dir / f"{model_name}_epillid_best.pt"
        row, y_true, y_pred = _evaluate_single_model(
            model_name=model_name,
            checkpoint_path=ckpt,
            loader=loader,
            dataset_class_to_idx=dataset.class_to_idx,
            device=device,
        )
        eval_rows.append(row)
        pred_cache[model_name] = (y_true, y_pred)
        _stage(3, total_stages, f"Da danh gia {model_name}: acc={row.accuracy:.4f}, macro_f1={row.macro_f1:.4f}")

    ensemble_row, y_true_ens, y_pred_ens = _evaluate_ensemble(
        model_names=trained_models,
        models_dir=models_dir,
        loader=loader,
        dataset_class_to_idx=dataset.class_to_idx,
        device=device,
    )
    eval_rows.append(ensemble_row)
    pred_cache[ensemble_row.model] = (y_true_ens, y_pred_ens)
    _stage(
        3,
        total_stages,
        f"Da danh gia {ensemble_row.model}: acc={ensemble_row.accuracy:.4f}, macro_f1={ensemble_row.macro_f1:.4f}",
    )

    # Stage 4: export report table + visual analysis.
    _stage(4, total_stages, "Xuat bao cao CSV/JSON va bieu do")
    summary_csv = report_dir / "evaluation_summary.csv"
    summary_json = report_dir / "evaluation_summary.json"
    score_chart = report_dir / "evaluation_comparison.png"

    _write_summary_csv(eval_rows, summary_csv)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in eval_rows], f, indent=2, ensure_ascii=False)

    _plot_score_comparison(eval_rows, score_chart)

    best_single = max(
        [r for r in eval_rows if r.model != ensemble_row.model],
        key=lambda r: (r.accuracy, r.macro_f1),
    )
    y_true_best, y_pred_best = pred_cache[best_single.model]
    _plot_confusion_matrix(
        y_true_best,
        y_pred_best,
        output_path=report_dir / f"confusion_matrix_{best_single.model}.png",
        title=f"Confusion Matrix - {best_single.model}",
    )
    _plot_confusion_matrix(
        y_true_ens,
        y_pred_ens,
        output_path=report_dir / "confusion_matrix_ensemble_weighted.png",
        title="Confusion Matrix - ensemble_weighted",
    )

    finished_at_dt = datetime.now()
    finished_at = finished_at_dt.isoformat(timespec="seconds")
    elapsed = (finished_at_dt - started_at_dt).total_seconds()

    # Stage 5: print final summary.
    _stage(5, total_stages, "Pipeline da hoan tat thanh cong")
    print(f"[KET_QUA] Thu muc du lieu: {data_dir}", flush=True)
    print(f"[KET_QUA] Mo hinh da train: {', '.join(trained_models)}", flush=True)
    print(f"[KET_QUA] Mo hinh don tot nhat: {best_single.model}", flush=True)
    print(f"[KET_QUA] Mo hinh ensemble: {ensemble_row.model}", flush=True)
    print(f"[KET_QUA] File tong hop CSV: {summary_csv}", flush=True)
    print(f"[KET_QUA] File tong hop JSON: {summary_json}", flush=True)
    print(f"[KET_QUA] Bieu do so sanh: {score_chart}", flush=True)
    print(f"[KET_QUA] Thu muc bao cao: {report_dir}", flush=True)

    return PipelineSummary(
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=elapsed,
        data_dir=data_dir,
        test_dir=test_dir,
        models_dir=str(models_dir),
        report_dir=str(report_dir),
        trained_models=trained_models,
        best_model=best_single.model,
        ensemble_model_name=ensemble_row.model,
    )


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
