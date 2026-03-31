from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Subset

from ..data.features import PillImageDataset, build_transforms
from ..models.model_factory import load_checkpoint, load_checkpoint_class_to_idx
from ..utils.model_paths import model_artifact_dir, resolve_model_checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Danh gia hieu nang cac mo hinh da huan luyen va xuat bieu do so sanh.")
    parser.add_argument("--data-dir", type=str, default="data_aligned", help="Data root with test split")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory of .pt checkpoints")
    parser.add_argument(
        "--model-list",
        type=str,
        default="resnet50,efficientnet_b0,vit_b_16",
        help="Comma-separated model names",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=0, help="Evaluation batch size (0 = auto)")
    parser.add_argument("--num-workers", type=int, default=0, help="Evaluation DataLoader workers")
    parser.add_argument("--max-samples", type=int, default=0, help="Max test samples for quick eval (0 = full)")
    parser.add_argument("--output", type=str, default="models/evaluation_summary.csv")
    parser.add_argument("--chart", type=str, default="models/evaluation_comparison.png")
    parser.add_argument(
        "--report-dir",
        type=str,
        default="models/reports/latest",
        help="Directory for JSON summary and confusion matrices",
    )
    return parser.parse_args()


def _evaluate_one_model(
    model_name: str,
    checkpoint_path: Path,
    loader: DataLoader,
    dataset_class_to_idx: Dict[str, int],
    device: torch.device,
) -> Tuple[Dict[str, object], List[str], List[str]]:
    ckpt_class_to_idx = load_checkpoint_class_to_idx(str(checkpoint_path), map_location=device)
    num_classes = len(ckpt_class_to_idx) if ckpt_class_to_idx else len(dataset_class_to_idx)

    model = load_checkpoint(
        model_name=model_name,
        num_classes=num_classes,
        checkpoint_path=str(checkpoint_path),
        map_location=device,
    ).to(device)
    model.eval()

    if ckpt_class_to_idx:
        inv_pred = {v: k for k, v in ckpt_class_to_idx.items()}
    else:
        inv_pred = {v: k for k, v in dataset_class_to_idx.items()}
    inv_true = {v: k for k, v in dataset_class_to_idx.items()}

    y_true_names: List[str] = []
    y_pred_names: List[str] = []

    with torch.inference_mode():
        for images, labels, _paths in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            true_batch = labels.tolist()
            pred_batch = preds.tolist()
            y_true_names.extend(inv_true.get(int(t), f"class_{int(t)}") for t in true_batch)
            y_pred_names.extend(inv_pred.get(int(p), f"class_{int(p)}") for p in pred_batch)

    acc = accuracy_score(y_true_names, y_pred_names)
    macro_f1 = f1_score(y_true_names, y_pred_names, average="macro", zero_division=0)

    result = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "num_samples": float(len(y_true_names)),
        "checkpoint": _portable_path(checkpoint_path),
    }
    return result, y_true_names, y_pred_names


def _write_summary_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "accuracy", "macro_f1", "num_samples", "checkpoint"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _plot_comparison(rows: List[Dict[str, object]], chart_path: Path) -> None:
    chart_path.parent.mkdir(parents=True, exist_ok=True)

    models = [str(r["model"]) for r in rows]
    accs = [float(r["accuracy"]) for r in rows]
    f1s = [float(r["macro_f1"]) for r in rows]

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10, 5))

    x = list(range(len(models)))
    width = 0.36
    ax.bar([i - width / 2 for i in x], accs, width=width, label="Accuracy")
    ax.bar([i + width / 2 for i in x], f1s, width=width, label="Macro F1")

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Evaluation Comparison")
    ax.legend(loc="best")

    for i, v in enumerate(accs):
        ax.text(i - width / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    for i, v in enumerate(f1s):
        ax.text(i + width / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _plot_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
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

    if len(labels) <= 18:
        threshold = cm.max() / 2.0 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    str(int(cm[i, j])),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > threshold else "black",
                    fontsize=8,
                )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_summary_json(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def _print_checkpoint_diagnostics(models_dir: Path, model_names: List[str]) -> None:
    print("[EVAL][ERROR] No valid checkpoints found for evaluation")
    print(f"[EVAL][INFO] models_dir={models_dir}")
    print("[EVAL][INFO] Expected checkpoint locations:")
    for model_name in model_names:
        expected = model_artifact_dir(models_dir, model_name) / f"{model_name}_epillid_best.pt"
        print(f"  - {model_name}: {expected}")

    found = sorted(models_dir.glob("**/*_epillid_best.pt")) if models_dir.exists() else []
    if found:
        print("[EVAL][INFO] Found candidate checkpoints under current models_dir:")
        for path in found[:15]:
            print(f"  - {path}")
        if len(found) > 15:
            print(f"  ... and {len(found) - 15} more")
    else:
        print("[EVAL][INFO] No *_epillid_best.pt found under current models_dir")


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_dir)
    test_root = data_root / "test"
    models_dir = Path(args.models_dir)

    if not test_root.exists():
        raise FileNotFoundError(f"Test split not found: {test_root}")

    model_names = [m.strip() for m in args.model_list.split(",") if m.strip()]
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA khong kha dung, tu chuyen sang CPU cho evaluate.")
        args.device = "cpu"
    device = torch.device(args.device)

    base_dataset = PillImageDataset(str(test_root), transform=build_transforms(train=False))
    eval_dataset = base_dataset
    max_samples = max(0, int(args.max_samples))
    if max_samples > 0 and len(base_dataset) > max_samples:
        eval_dataset = Subset(base_dataset, range(max_samples))

    batch_size = int(args.batch_size)
    if batch_size <= 0:
        batch_size = 64 if device.type == "cpu" else 32
    batch_size = max(1, batch_size)

    num_workers = max(0, int(args.num_workers))
    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    print(
        f"[EVAL] device={device.type} batch_size={batch_size} num_workers={num_workers} "
        f"samples={len(eval_dataset)}/{len(base_dataset)}"
    )

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    pred_cache: Dict[str, Tuple[List[str], List[str]]] = {}
    for model_name in model_names:
        ckpt = resolve_model_checkpoint_path(base_output_dir=models_dir, model_name=model_name)
        if ckpt is None:
            continue

        print(f"[EVAL] Using checkpoint for {model_name}: {ckpt}")

        scores, y_true, y_pred = _evaluate_one_model(
            model_name=model_name,
            checkpoint_path=ckpt,
            loader=loader,
            dataset_class_to_idx=base_dataset.class_to_idx,
            device=device,
        )
        row: Dict[str, object] = {
            "model": model_name,
            "accuracy": scores["accuracy"],
            "macro_f1": scores["macro_f1"],
            "num_samples": scores["num_samples"],
            "checkpoint": scores["checkpoint"],
        }
        rows.append(row)
        pred_cache[model_name] = (y_true, y_pred)

    if not rows:
        _print_checkpoint_diagnostics(models_dir=models_dir, model_names=model_names)
        raise RuntimeError("No valid checkpoints found for evaluation")

    summary_csv = Path(args.output)
    summary_chart = Path(args.chart)
    summary_json = report_dir / "evaluation_summary.json"

    _write_summary_csv(rows, summary_csv)
    _plot_comparison(rows, summary_chart)
    _write_summary_json(rows, summary_json)

    for model_name, (y_true, y_pred) in pred_cache.items():
        _plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            output_path=report_dir / f"confusion_matrix_{model_name}.png",
            title=f"Confusion Matrix - {model_name}",
        )

    print(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved chart: {summary_chart}")
    print(f"Saved summary JSON: {summary_json}")
    print(f"Saved confusion matrices to: {report_dir}")


if __name__ == "__main__":
    main()
