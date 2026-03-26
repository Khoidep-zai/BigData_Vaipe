from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from ..data.features import PillImageDataset, build_transforms
from ..models.model_factory import load_checkpoint, load_checkpoint_class_to_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Đánh giá hiệu năng các mô hình đã huấn luyện và xuất biểu đồ so sánh.")
    parser.add_argument("--data-dir", type=str, default="data_aligned", help="Data root with test split")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory of .pt checkpoints")
    parser.add_argument(
        "--model-list",
        type=str,
        default="efficientnet_b0,resnet50",
        help="Comma-separated model names",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="models/evaluation_summary.csv")
    parser.add_argument("--chart", type=str, default="models/evaluation_comparison.png")
    return parser.parse_args()


def _evaluate_one_model(
    model_name: str,
    checkpoint_path: Path,
    loader: DataLoader,
    dataset_class_to_idx: Dict[str, int],
    device: torch.device,
) -> Dict[str, float]:
    # Ưu tiên lấy bảng ánh xạ lớp (class_to_idx) từ checkpoint để tránh lệch nhãn khi dự đoán.
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

    # Dùng tên lớp (string) thay vì chỉ số (int) để tính toán, giúp kết quả chính xác ngay cả khi thứ tự lớp trong các mô hình khác nhau.
    with torch.no_grad():
        for images, labels, _paths in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            for t, p in zip(labels.tolist(), preds.tolist()):
                y_true_names.append(inv_true.get(int(t), f"class_{int(t)}"))
                y_pred_names.append(inv_pred.get(int(p), f"class_{int(p)}"))

    acc = accuracy_score(y_true_names, y_pred_names)
    macro_f1 = f1_score(y_true_names, y_pred_names, average="macro", zero_division=0)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "num_samples": float(len(y_true_names)),
    }


def _write_summary_csv(rows: List[Dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "accuracy", "macro_f1", "num_samples"])
        writer.writeheader()
        writer.writerows(rows)


def _plot_comparison(rows: List[Dict[str, float]], chart_path: Path) -> None:
    chart_path.parent.mkdir(parents=True, exist_ok=True)

    models = [r["model"] for r in rows]
    accs = [float(r["accuracy"]) for r in rows]
    f1s = [float(r["macro_f1"]) for r in rows]

    # Vẽ biểu đồ cột so sánh Accuracy và F1-Score giữa các mô hình để dễ dàng chọn ra mô hình tốt nhất.
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


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_dir)
    test_root = data_root / "test"
    models_dir = Path(args.models_dir)

    if not test_root.exists():
        raise FileNotFoundError(f"Test split not found: {test_root}")

    dataset = PillImageDataset(str(test_root), transform=build_transforms(train=False))
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    model_names = [m.strip() for m in args.model_list.split(",") if m.strip()]
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA không khả dụng, tự chuyển sang CPU cho evaluate.")
        args.device = "cpu"
    device = torch.device(args.device)

    rows: List[Dict[str, float]] = []
    for model_name in model_names:
        ckpt = models_dir / f"{model_name}_epillid_best.pt"
        if not ckpt.exists():
            continue

        scores = _evaluate_one_model(
            model_name=model_name,
            checkpoint_path=ckpt,
            loader=loader,
            dataset_class_to_idx=dataset.class_to_idx,
            device=device,
        )
        row: Dict[str, float] = {
            "model": model_name,
            "accuracy": scores["accuracy"],
            "macro_f1": scores["macro_f1"],
            "num_samples": scores["num_samples"],
        }
        rows.append(row)

    if not rows:
        raise RuntimeError("No valid checkpoints found for evaluation")

    _write_summary_csv(rows, Path(args.output))
    _plot_comparison(rows, Path(args.chart))

    print(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"Saved summary: {args.output}")
    print(f"Saved chart: {args.chart}")


if __name__ == "__main__":
    main()
