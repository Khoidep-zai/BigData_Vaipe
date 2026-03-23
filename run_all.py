#!/usr/bin/env python
"""
🚀 THUOC - SETUP & RUN CHÍNH (All-in-One)

Tương ứng với: train_optimized.py + evaluate_report + visualization

Sử dụng:
    python run_all.py                    # Run all 3 models + eval
    python run_all.py --model resnet50   # Run 1 model
    python run_all.py --compare-only     # Chỉ compare kết quả
    python run_all.py --data-dir data_aligned  # Custom data dir

Tính năng:
    ✓ Train 3 models với optimal hyperparams
    ✓ Evaluate & confusion matrix
    ✓ Ensemble weighted
    ✓ Summary CSV + JSON
    ✓ Visualization

Chạy trên local: python run_all.py --device cuda
Chạy trên Colab: python run_all.py --device cuda (sẽ tự detect GPU)
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from optimal_configs import OPTIMAL_CONFIGS


def _class_dirs(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {p.name for p in root.iterdir() if p.is_dir()}


def _has_consistent_splits(data_dir: Path) -> bool:
    train_classes = _class_dirs(data_dir / "train")
    val_classes = _class_dirs(data_dir / "val")
    test_classes = _class_dirs(data_dir / "test")
    if not train_classes or not val_classes or not test_classes:
        return False
    return train_classes == val_classes == test_classes


def run_cmd(cmd: List[str], description: str = "") -> bool:
    """Chạy command và báo cáo lỗi"""
    if description:
        print(f"\n{'='*70}")
        print(f"▶ {description}")
        print(f"{'='*70}")
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def train_models(
    data_dir: str = "data",
    models: List[str] = None,
    device: str = "cuda",
    batch_size: int = 16,
    seed: int = 42,
    max_train_val_gap: float = 0.06,
    freeze_backbone_epochs: int = 5,
    output_dir: str = "models",
) -> bool:
    """Train Models"""
    if models is None:
        models = ["resnet50", "efficientnet_b0", "vit_b_16"]
    
    success_count = 0
    
    for model_name in models:
        config = OPTIMAL_CONFIGS.get(model_name)
        if not config:
            print(f"[SKIP] Không có config cho {model_name}")
            continue

        model_gap = float(config.get("max_train_val_gap", max_train_val_gap))
        model_freeze_epochs = int(config.get("freeze_backbone_epochs", freeze_backbone_epochs))
        
        cmd = [
            sys.executable,
            "train_cli.py",
            "--mode",
            "single",
            "--model",
            model_name,
            "--data-dir",
            data_dir,
            "--epochs",
            str(config.get("epochs", 25)),
            "--batch-size",
            str(batch_size),
            "--lr",
            str(config["lr"]),
            "--weight-decay",
            str(config["weight_decay"]),
            "--label-smoothing",
            str(config["label_smoothing"]),
            "--mixup-alpha",
            str(config["mixup_alpha"]),
            "--early-stop-patience",
            str(config.get("early_stop_patience", 6)),
            "--num-workers",
            "0",  # Windows/Colab friendly
            "--seed",
            str(seed),
            "--max-train-val-gap",
            str(model_gap),
            "--freeze-backbone-epochs",
            str(model_freeze_epochs),
            "--device",
            device,
            "--output-dir",
            output_dir,
        ]
        
        success = run_cmd(cmd, f"Training {model_name}")
        if success:
            success_count += 1
            print(f"✓ {model_name} trained successfully")
        else:
            print(f"✗ {model_name} training failed")
    
    return success_count == len(models)


def evaluate_models(
    data_dir: str = "data",
    models: List[str] = None,
    device: str = "cuda",
    output_dir: str = "models",
) -> bool:
    """Evaluate & Generate Reports"""
    if models is None:
        models = ["resnet50", "efficientnet_b0", "vit_b_16"]
    
    model_list_str = ",".join(models)
    
    cmd = [
        sys.executable,
        "-m",
        "src.evaluate_report",
        "--data-dir",
        data_dir,
        "--models-dir",
        output_dir,
        "--model-list",
        model_list_str,
        "--device",
        device,
        "--output",
        f"{output_dir}/evaluation_summary.csv",
        "--chart",
        f"{output_dir}/evaluation_comparison.png",
    ]
    
    return run_cmd(cmd, "Evaluating Models & Generating Reports")


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_metric(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def _load_json_file(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _load_eval_map(summary_csv: Path) -> Dict[str, Dict[str, float]]:
    eval_map: Dict[str, Dict[str, float]] = {}
    if not summary_csv.exists():
        return eval_map

    try:
        with open(summary_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_name = (row.get("model") or "").strip()
                if not model_name:
                    continue
                eval_map[model_name] = {
                    "accuracy": _safe_float(row.get("accuracy")) or 0.0,
                    "macro_f1": _safe_float(row.get("macro_f1")) or 0.0,
                }
    except Exception:
        return {}

    return eval_map


def _print_table(headers: List[str], rows: List[List[str]]) -> None:
    if not rows:
        return
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _line(values: List[str]) -> str:
        return " | ".join(v.ljust(widths[i]) for i, v in enumerate(values))

    sep = "-+-".join("-" * w for w in widths)
    print(_line(headers))
    print(sep)
    for row in rows:
        print(_line(row))


def export_training_tables(models: List[str], output_dir: str = "models") -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    eval_map = _load_eval_map(output_root / "evaluation_summary.csv")
    rows: List[Dict[str, str]] = []

    for model in models:
        history = _load_json_file(output_root / f"{model}_epillid_history.json")
        metrics = _load_json_file(output_root / f"{model}_epillid_best.metrics.json")

        epochs = history.get("epoch", []) if isinstance(history.get("epoch", []), list) else []
        train_acc = history.get("train_acc", []) if isinstance(history.get("train_acc", []), list) else []
        val_acc = history.get("val_acc", []) if isinstance(history.get("val_acc", []), list) else []

        n_epochs = len(epochs)
        last_train = _safe_float(train_acc[-1]) if train_acc else None
        last_val = _safe_float(val_acc[-1]) if val_acc else None
        final_gap = (last_train - last_val) if (last_train is not None and last_val is not None) else None

        best_val_hist = None
        best_epoch_hist = None
        if val_acc and epochs and len(val_acc) == len(epochs):
            valid_indices = [i for i in range(len(val_acc)) if _safe_float(val_acc[i]) is not None]
            if valid_indices:
                best_idx = max(valid_indices, key=lambda i: _safe_float(val_acc[i]) or float("-inf"))
                best_val_hist = _safe_float(val_acc[best_idx])
                if best_val_hist is not None and best_idx < len(epochs):
                    best_epoch_hist = int(epochs[best_idx])

        best_val_metric = _safe_float(metrics.get("best_val_acc"))
        best_val = best_val_metric if best_val_metric is not None else best_val_hist
        best_epoch_value = metrics.get("best_val_epoch")
        if best_epoch_value is None:
            best_epoch_value = metrics.get("epochs")
        if best_epoch_value is None:
            best_epoch_value = best_epoch_hist
        best_epoch = int(best_epoch_value) if best_epoch_value is not None else 0

        eval_item = eval_map.get(model, {})
        eval_acc = _safe_float(eval_item.get("accuracy"))
        eval_f1 = _safe_float(eval_item.get("macro_f1"))

        rows.append(
            {
                "model": model,
                "epochs_ran": str(n_epochs),
                "best_epoch": str(best_epoch),
                "best_val_acc": _format_metric(best_val),
                "final_train_acc": _format_metric(last_train),
                "final_val_acc": _format_metric(last_val),
                "final_gap": _format_metric(final_gap),
                "eval_accuracy": _format_metric(eval_acc),
                "eval_macro_f1": _format_metric(eval_f1),
            }
        )

    fieldnames = [
        "model",
        "epochs_ran",
        "best_epoch",
        "best_val_acc",
        "final_train_acc",
        "final_val_acc",
        "final_gap",
        "eval_accuracy",
        "eval_macro_f1",
    ]

    csv_path = output_root / "training_results_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_root / "training_results_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| Model | Epochs Ran | Best Epoch | Best Val Acc | Final Train Acc | Final Val Acc | Final Gap | Eval Accuracy | Eval Macro F1 |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['model']} | {row['epochs_ran']} | {row['best_epoch']} | {row['best_val_acc']} | {row['final_train_acc']} | {row['final_val_acc']} | {row['final_gap']} | {row['eval_accuracy']} | {row['eval_macro_f1']} |\n"
            )

    print(f"\n{'='*70}")
    print("📋 BẢNG TỔNG HỢP KẾT QUẢ TRAIN")
    print(f"{'='*70}")
    headers = [
        "Model",
        "Epochs",
        "BestEp",
        "BestVal",
        "FinalTr",
        "FinalVal",
        "Gap",
        "EvalAcc",
        "EvalF1",
    ]
    table_rows = [
        [
            r["model"],
            r["epochs_ran"],
            r["best_epoch"],
            r["best_val_acc"],
            r["final_train_acc"],
            r["final_val_acc"],
            r["final_gap"],
            r["eval_accuracy"],
            r["eval_macro_f1"],
        ]
        for r in rows
    ]
    _print_table(headers, table_rows)
    print(f"Saved table CSV: {csv_path}")
    print(f"Saved table MD:  {md_path}")


def display_results(output_dir: str = "models") -> None:
    """Hiển thị kết quả"""
    summary_csv = Path(output_dir) / "evaluation_summary.csv"
    
    print(f"\n{'='*70}")
    print("📊 KẾT QUẢ TRAINING")
    print(f"{'='*70}\n")
    
    if summary_csv.exists():
        with open(summary_csv, 'r', encoding='utf-8') as f:
            print(f.read())
    else:
        print("[WARN] Không tìm thấy evaluation_summary.csv")
    
    print(f"\n{'='*70}")
    print("📁 Output Files:")
    print(f"{'='*70}")
    print(f"  CSV:           {output_dir}/evaluation_summary.csv")
    print(f"  JSON:          {output_dir}/evaluation_summary.json")
    print(f"  Chart:         {output_dir}/evaluation_comparison.png")
    print(f"  Train Table:   {output_dir}/training_results_table.csv")
    print(f"  Train TableMD: {output_dir}/training_results_table.md")
    print(f"  Report Dir:    {output_dir}/reports/latest/")
    print(f"  Training Curves: {output_dir}/*_epillid_training_curves.png")
    print(f"  Confusion Matrix: {output_dir}/reports/latest/confusion_matrix_*.png")


def main():
    parser = argparse.ArgumentParser(
        description="THUOC - All-in-One Setup & Run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py                    # Run all (train + eval)
  python run_all.py --model resnet50   # Run 1 model
  python run_all.py --compare-only     # Just compare results
  python run_all.py --data-dir data_aligned   # Custom data dir
  python run_all.py --device cpu       # CPU mode (slow)
        """,
    )
    
    parser.add_argument(
        "--model",
        choices=["resnet50", "efficientnet_b0", "vit_b_16", "all"],
        default="all",
        help="Model(s) to train (default: all)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data_aligned",
        help="Data directory with train/val/test (default: data_aligned)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device (default: cuda)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/val curves (default: 42)",
    )
    parser.add_argument(
        "--max-train-val-gap",
        type=float,
        default=0.12,
        help="Stop early when train-val accuracy gap stays above this threshold (default: 0.12)",
    )
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=3,
        help="Freeze backbone first N epochs to reduce train/val divergence on small data (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory (default: models)",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only evaluate existing checkpoints, don't train",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation after training",
    )
    
    args = parser.parse_args()

    # Validate data dir
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    if not (data_dir / "train").exists():
        print(f"[ERROR] Missing train folder in {args.data_dir}")
        sys.exit(1)

    if not _has_consistent_splits(data_dir):
        fallback_dir = Path("data_aligned")
        if args.data_dir != "data_aligned" and _has_consistent_splits(fallback_dir):
            print(
                f"[WARN] Class train/val/test không đồng nhất ở {args.data_dir}. "
                "Tự chuyển sang data_aligned."
            )
            data_dir = fallback_dir
            args.data_dir = str(fallback_dir)
        else:
            print(
                f"[ERROR] Class train/val/test không đồng nhất ở {args.data_dir}. "
                "Hãy dùng --data-dir data_aligned"
            )
            sys.exit(1)
    
    models = ["resnet50", "efficientnet_b0", "vit_b_16"] if args.model == "all" else [args.model]
    
    print(f"\n{'='*70}")
    print("🚀 THUOC - ALL-IN-ONE SETUP")
    print(f"{'='*70}")
    print(f"Models:       {', '.join(models)}")
    print(f"Data:         {args.data_dir}")
    print(f"Device:       {args.device}")
    print(f"Batch Size:   {args.batch_size}")
    print(f"Seed:         {args.seed}")
    print(f"Max Gap:      {args.max_train_val_gap}")
    print(f"Freeze BB:    {args.freeze_backbone_epochs}")
    print(f"Output:       {args.output_dir}")
    
    # Train
    if not args.compare_only:
        success = train_models(
            data_dir=str(args.data_dir),
            models=models,
            device=args.device,
            batch_size=args.batch_size,
            seed=args.seed,
            max_train_val_gap=args.max_train_val_gap,
            freeze_backbone_epochs=args.freeze_backbone_epochs,
            output_dir=args.output_dir,
        )
        
        if not success:
            print("\n[WARN] Some models failed to train")
    
    # Evaluate
    if not args.skip_eval:
        evaluate_models(
            data_dir=str(args.data_dir),
            models=models,
            device=args.device,
            output_dir=args.output_dir,
        )

    # Export training/evaluation summary table after all runs.
    export_training_tables(models=models, output_dir=args.output_dir)
    
    # Display
    display_results(args.output_dir)
    
    print(f"\n{'='*70}")
    print("✓ DONE!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
