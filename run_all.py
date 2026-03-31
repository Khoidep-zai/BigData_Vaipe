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

from Review.optimal_configs import OPTIMAL_CONFIGS
from src.data.data_setup import discover_or_prepare_data_dir, prepare_metadata_artifacts
from src.utils.model_paths import (
    model_artifact_dir,
    resolve_model_artifact_path,
    resolve_model_checkpoint_path,
)
from src.utils.runtime_artifacts import ensure_runtime_dirs, mirror_artifacts_to_runtime_dirs


def _class_dirs(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {p.name for p in root.iterdir() if p.is_dir()}


def _has_consistent_splits(data_dir: Path) -> bool:
    # Đảm bảo rằng thư mục train, val và test đều có cùng danh sách các lớp (thư mục con) thuốc giống nhau trước khi bắt đầu huấn luyện.
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


def _resolve_runtime_device(requested_device: str) -> str:
    if requested_device == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    print("[WARN] CUDA không khả dụng ở runtime, chuyển sang CPU.")
    return "cpu"


def train_models(
    data_dir: str = "data",
    models: List[str] = None,
    device: str = "cuda",
    batch_size: int = 16,
    seed: int = 42,
    max_train_val_gap: float = 0.06,
    freeze_backbone_epochs: int = 5,
    backbone_lr_scale: float = 0.2,
    ema_decay: float = 0.997,
    tta_views: int = 3,
    output_dir: str = "models",
    num_workers: int = 0,
    fast_train: bool = False,
    augment_profile: str = "default",
    vector_grayscale_prob: float = 0.0,
) -> bool:
    """Huấn luyện các mô hình được chọn, sử dụng bộ tham số tối ưu riêng cho từng loại kiến trúc (backbone)."""
    if models is None:
        models = ["resnet50", "efficientnet_b0", "vit_b_16"]
    
    success_count = 0
    cpu_mode = str(device).lower() == "cpu"
    
    for model_name in models:
        config = OPTIMAL_CONFIGS.get(model_name)
        if not config:
            print(f"[SKIP] Không có config cho {model_name}")
            continue

        model_gap = float(config.get("max_train_val_gap", max_train_val_gap))
        model_freeze_epochs = int(config.get("freeze_backbone_epochs", freeze_backbone_epochs))
        model_backbone_lr_scale = float(config.get("backbone_lr_scale", backbone_lr_scale))
        model_ema_decay = float(config.get("ema_decay", ema_decay))
        model_tta_views = int(config.get("tta_views", tta_views))
        model_epochs = int(config.get("epochs", 25))
        model_patience = int(config.get("early_stop_patience", 6))
        model_train_metric_every = 3
        model_quick_val_tta_views = 1
        model_full_tta_on_save = True

        if fast_train:
            model_epochs = max(6, int(round(model_epochs * 0.45)))
            model_tta_views = 1
            model_ema_decay = 0.0
            model_patience = min(model_patience, 4)
            model_freeze_epochs = min(model_freeze_epochs, 1)
            model_train_metric_every = 4
            model_full_tta_on_save = False

        if cpu_mode:
            # CPU ưu tiên throughput: giảm các lượt evaluate nặng và giới hạn chi phí mỗi epoch.
            model_train_metric_every = max(model_train_metric_every, 5)
            model_full_tta_on_save = False
            if not fast_train:
                model_epochs = max(10, int(round(model_epochs * 0.60)))
                model_tta_views = 1
                model_ema_decay = 0.0
                model_patience = min(model_patience, 5)
                model_freeze_epochs = min(model_freeze_epochs, 2)
        
        # Gọi lệnh chạy thực tế qua file train_cli.py ở chế độ đơn lẻ (single) để huấn luyện.
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
            str(model_epochs),
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
            "--augment-profile",
            str(augment_profile),
            "--vector-grayscale-prob",
            str(float(vector_grayscale_prob)),
            "--early-stop-patience",
            str(model_patience),
            "--num-workers",
            str(max(0, int(num_workers))),
            "--seed",
            str(seed),
            "--max-train-val-gap",
            str(model_gap),
            "--freeze-backbone-epochs",
            str(model_freeze_epochs),
            "--backbone-lr-scale",
            str(model_backbone_lr_scale),
            "--ema-decay",
            str(model_ema_decay),
            "--tta-views",
            str(model_tta_views),
            "--train-metric-every",
            str(model_train_metric_every),
            "--quick-val-tta-views",
            str(model_quick_val_tta_views),
            "--skip-metadata-artifacts",
            "--device",
            device,
            "--output-dir",
            output_dir,
        ]

        if model_full_tta_on_save:
            cmd.append("--full-tta-on-save")
        else:
            cmd.append("--no-full-tta-on-save")
        
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
    report_dir: str = "models/reports/latest",
    eval_batch_size: int = 0,
    eval_num_workers: int = 0,
    max_eval_samples: int = 0,
) -> bool:
    """Đánh giá các mô hình đã lưu (checkpoint) và xuất ra các file báo cáo, biểu đồ so sánh."""
    if models is None:
        models = ["resnet50", "efficientnet_b0", "vit_b_16"]
    
    model_list_str = ",".join(models)
    
    # Tái sử dụng module đánh giá có sẵn để giữ cho logic code tập trung ở một nơi.
    eval_root = _evaluation_results_dir(output_dir)
    eval_root.mkdir(parents=True, exist_ok=True)
    summary_csv = (eval_root / "evaluation_summary.csv").as_posix()
    summary_chart = (eval_root / "evaluation_comparison.png").as_posix()

    cmd = [
        sys.executable,
        "-m",
        "src.evaluation.evaluate_report",
        "--data-dir",
        data_dir,
        "--models-dir",
        output_dir,
        "--model-list",
        model_list_str,
        "--device",
        device,
        "--output",
        summary_csv,
        "--chart",
        summary_chart,
        "--report-dir",
        report_dir,
        "--batch-size",
        str(int(eval_batch_size)),
        "--num-workers",
        str(max(0, int(eval_num_workers))),
        "--max-samples",
        str(max(0, int(max_eval_samples))),
    ]
    
    return run_cmd(cmd, "Evaluating Models & Generating Reports")


def _find_expected_checkpoints(models: List[str], output_dir: str) -> Dict[str, Path]:
    return {
        m: model_artifact_dir(output_dir, m) / f"{m}_epillid_best.pt"
        for m in models
    }


def _resolve_checkpoint_path(model_name: str, output_dir: str) -> Optional[Path]:
    return resolve_model_checkpoint_path(base_output_dir=output_dir, model_name=model_name)


def _print_missing_checkpoint_help(models: List[str], output_dir: str) -> None:
    expected = _find_expected_checkpoints(models, output_dir)
    missing = {m: p for m, p in expected.items() if _resolve_checkpoint_path(m, output_dir) is None}
    if not missing:
        return

    print("[ERROR] --compare-only nhung khong tim thay checkpoint hop le:")
    for model_name, ckpt_path in missing.items():
        print(f"  - {model_name}: {ckpt_path}")

    output_root = Path(output_dir)
    nested_candidates = sorted(output_root.glob("**/*_epillid_best.pt"))
    if nested_candidates:
        print("[HINT] Da tim thay checkpoint o thu muc con (co the khong dung --output-dir hien tai):")
        for candidate in nested_candidates[:10]:
            print(f"  - {candidate}")
        if len(nested_candidates) > 10:
            print(f"  ... va {len(nested_candidates) - 10} file khac")

    print("[FIX] Chon 1 trong 2 cach:")
    print("  1) Train truoc: python run_all.py")
    print(f"  2) Chi ro output dung: python run_all.py --compare-only --output-dir <dir_chua_checkpoint>")


def _has_required_checkpoints(models: List[str], output_dir: str) -> bool:
    return all(_resolve_checkpoint_path(model_name=m, output_dir=output_dir) is not None for m in models)


def _default_report_dir(output_dir: str) -> str:
    return str(Path(output_dir) / "reports" / "latest")


def _training_results_dir(output_dir: str) -> Path:
    return Path(output_dir) / "results" / "training"


def _evaluation_results_dir(output_dir: str) -> Path:
    return Path(output_dir) / "results" / "evaluation"


def _normalize_model_artifacts(output_dir: str, model_names: List[str]) -> int:
    """Move legacy root-level model artifacts into canonical models/AI/<model_alias>/ dirs."""
    root = Path(output_dir)
    moved_count = 0

    for model_name in model_names:
        dst_dir = model_artifact_dir(root, model_name)
        dst_dir.mkdir(parents=True, exist_ok=True)

        legacy_names = [
            f"{model_name}_epillid_best.pt",
            f"{model_name}_epillid_best.metrics.json",
            f"{model_name}_epillid_history.json",
            f"{model_name}_training_curves.png",
            f"{model_name}_epillid_training_curves.png",
        ]

        for legacy_name in legacy_names:
            src = root / legacy_name
            if not src.exists() or not src.is_file():
                continue

            target_name = legacy_name
            if legacy_name.endswith("_epillid_training_curves.png"):
                target_name = f"{model_name}_training_curves.png"

            dst = dst_dir / target_name
            if dst.exists():
                try:
                    if src.stat().st_mtime > dst.stat().st_mtime:
                        dst.unlink()
                        src.replace(dst)
                        moved_count += 1
                    else:
                        src.unlink()
                        moved_count += 1
                except Exception:
                    continue
            else:
                try:
                    src.replace(dst)
                    moved_count += 1
                except Exception:
                    continue

    return moved_count


def _normalize_evaluation_artifacts(output_dir: str) -> int:
    """Move legacy root-level evaluation artifacts into models/results/evaluation."""
    root = Path(output_dir)
    dst_dir = _evaluation_results_dir(output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    moved_count = 0

    # Migrate typo folder (evalution -> evaluation) from older runs.
    typo_dir = root / "results" / "evalution"
    if typo_dir.exists() and typo_dir.is_dir():
        for name in ["evaluation_summary.csv", "evaluation_comparison.png"]:
            src = typo_dir / name
            if not src.exists() or not src.is_file():
                continue

            dst = dst_dir / name
            if dst.exists():
                try:
                    if src.stat().st_mtime > dst.stat().st_mtime:
                        dst.unlink()
                        src.replace(dst)
                        moved_count += 1
                    else:
                        src.unlink()
                        moved_count += 1
                except Exception:
                    continue
            else:
                try:
                    src.replace(dst)
                    moved_count += 1
                except Exception:
                    continue

        try:
            if not any(typo_dir.iterdir()):
                typo_dir.rmdir()
        except Exception:
            pass

    for name in ["evaluation_summary.csv", "evaluation_comparison.png"]:
        src = root / name
        if not src.exists() or not src.is_file():
            continue

        dst = dst_dir / name
        if dst.exists():
            try:
                if src.stat().st_mtime > dst.stat().st_mtime:
                    dst.unlink()
                    src.replace(dst)
                    moved_count += 1
                else:
                    src.unlink()
                    moved_count += 1
            except Exception:
                continue
        else:
            try:
                src.replace(dst)
                moved_count += 1
            except Exception:
                continue

    return moved_count


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(str(value))
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
    training_root = _training_results_dir(output_dir)
    training_root.mkdir(parents=True, exist_ok=True)

    eval_map = _load_eval_map(_evaluation_results_dir(output_dir) / "evaluation_summary.csv")
    rows: List[Dict[str, str]] = []

    for model in models:
        history_path = resolve_model_artifact_path(
            base_output_dir=output_root,
            model_name=model,
            suffix="_epillid_history.json",
        )
        metrics_path = resolve_model_artifact_path(
            base_output_dir=output_root,
            model_name=model,
            suffix="_epillid_best.metrics.json",
        )
        history = _load_json_file(history_path) if history_path is not None else {}
        metrics = _load_json_file(metrics_path) if metrics_path is not None else {}

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

    csv_path = training_root / "training_results_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = training_root / "training_results_table.md"
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


def display_results(output_dir: str = "models", report_dir: str = "models/reports/latest") -> None:
    """Hiển thị kết quả"""
    evaluation_root = _evaluation_results_dir(output_dir)
    summary_csv = evaluation_root / "evaluation_summary.csv"
    training_root = _training_results_dir(output_dir)
    
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
    print(f"  CSV:           {evaluation_root / 'evaluation_summary.csv'}")
    print(f"  JSON:          {report_dir}/evaluation_summary.json")
    print(f"  Chart:         {evaluation_root / 'evaluation_comparison.png'}")
    print(f"  Train Table:   {training_root / 'training_results_table.csv'}")
    print(f"  Train TableMD: {training_root / 'training_results_table.md'}")
    print(f"  Report Dir:    {report_dir}")
    output_root = Path(output_dir)
    curves_root = output_root / "AI" if output_root.name != "AI" else output_root
    print(f"  Training Curves: {curves_root.as_posix()}/*/*_training_curves.png")
    print(f"  Confusion Matrix: {report_dir}/confusion_matrix_*.png")


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
    python run_all.py --device cpu       # CPU mode (auto optimized)
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
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (default: 0 on Windows for stability)",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=0,
        help="Evaluation batch size (0 = auto, often faster on CPU)",
    )
    parser.add_argument(
        "--eval-num-workers",
        type=int,
        default=0,
        help="Evaluation DataLoader workers (default: 0)",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=0,
        help="Evaluate only first N test samples for quick iteration (0 = full test)",
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
        "--report-dir",
        type=str,
        default="",
        help="Directory for evaluation JSON/confusion artifacts (default: <output-dir>/reports/latest)",
    )
    parser.add_argument("--log-dir", type=str, default="log", help="Runtime folder for log files")
    parser.add_argument("--json-dir", type=str, default="json", help="Runtime folder for json files")
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
    parser.add_argument(
        "--fast-train",
        action="store_true",
        help="Use faster training preset (fewer epochs, no EMA, no val-TTA).",
    )
    parser.add_argument(
        "--cpu-optimize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically apply CPU speed optimizations when effective device is CPU (default: enabled).",
    )
    parser.add_argument(
        "--skip-metadata-artifacts",
        action="store_true",
        help="Skip metadata clean/vector export at startup.",
    )
    parser.add_argument(
        "--augment-profile",
        type=str,
        default="default",
        choices=["default", "lab6_stable"],
        help="Train augmentation profile for underlying train_cli runs.",
    )
    parser.add_argument(
        "--vector-grayscale-prob",
        type=float,
        default=0.0,
        help="Probability of grayscale augmentation during training (Lab6 vector-inspired).",
    )
    
    args = parser.parse_args()
    if not str(args.report_dir).strip():
        args.report_dir = _default_report_dir(args.output_dir)

    # Compare-only mode does not require metadata rebuild, skip it to reduce startup time.
    if args.compare_only and not args.skip_metadata_artifacts:
        args.skip_metadata_artifacts = True

    ensure_runtime_dirs(log_dir=args.log_dir, json_dir=args.json_dir)

    try:
        args.data_dir = discover_or_prepare_data_dir(preferred=args.data_dir, seed=int(args.seed))
    except Exception as exc:
        print(f"[ERROR] Khong the chuan bi du lieu train: {exc}")
        sys.exit(1)

    if not args.skip_metadata_artifacts:
        metadata_artifacts_obj = prepare_metadata_artifacts(data_root="data")
        if metadata_artifacts_obj and isinstance(metadata_artifacts_obj, dict):
            clean_summary = metadata_artifacts_obj.get("clean_summary", {})
            clean_csv = clean_summary.get("output_csv", "") if isinstance(clean_summary, dict) else ""
            vector_csv = metadata_artifacts_obj.get("vector_csv", "")
            print(f"[META] Clean metadata CSV: {clean_csv}")
            print(f"[META] Metadata vector CSV: {vector_csv}")

    data_dir = Path(args.data_dir)

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
    requested_device = args.device
    runtime_device = _resolve_runtime_device(requested_device)
    cpu_optimized = bool(args.cpu_optimize and runtime_device == "cpu")

    if cpu_optimized and not args.fast_train and not args.compare_only:
        args.fast_train = True
        print(
            "[CPU] Bật tự động chế độ train nhanh: giảm epochs, tắt EMA/TTA nặng để tăng tốc trên CPU. "
            "Dùng --no-cpu-optimize để tắt."
        )

    args.device = runtime_device

    if int(args.eval_batch_size) <= 0:
        args.eval_batch_size = max(32, int(args.batch_size) * 4) if args.device == "cpu" else max(16, int(args.batch_size))
    args.eval_num_workers = max(0, int(args.eval_num_workers))
    args.max_eval_samples = max(0, int(args.max_eval_samples))
    
    print(f"\n{'='*70}")
    print("🚀 THUOC - ALL-IN-ONE SETUP")
    print(f"{'='*70}")
    print(f"Models:       {', '.join(models)}")
    print(f"Data:         {args.data_dir}")
    print(f"Device Req:   {requested_device}")
    print(f"Device Run:   {args.device}")
    print(f"Batch Size:   {args.batch_size}")
    print(f"Workers:      {args.num_workers}")
    print(f"Eval Batch:   {args.eval_batch_size}")
    print(f"Eval Workers: {args.eval_num_workers}")
    print(f"Eval Samples: {'full' if args.max_eval_samples == 0 else args.max_eval_samples}")
    print(f"Seed:         {args.seed}")
    print(f"Max Gap:      {args.max_train_val_gap}")
    print(f"Freeze BB:    {args.freeze_backbone_epochs}")
    print(f"Output:       {args.output_dir}")
    print(f"Report Dir:   {args.report_dir}")
    print(f"Fast Train:   {args.fast_train}")
    print(f"CPU Optimize: {cpu_optimized}")
    
    # Train
    if not args.compare_only:
        success = train_models(
            data_dir=str(data_dir),
            models=models,
            device=args.device,
            batch_size=args.batch_size,
            seed=args.seed,
            max_train_val_gap=args.max_train_val_gap,
            freeze_backbone_epochs=args.freeze_backbone_epochs,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
            fast_train=args.fast_train,
            augment_profile=args.augment_profile,
            vector_grayscale_prob=float(args.vector_grayscale_prob),
        )
        
        if not success:
            print("\n[WARN] Some models failed to train")

    if args.compare_only and not _has_required_checkpoints(models=models, output_dir=args.output_dir):
        _print_missing_checkpoint_help(models=models, output_dir=args.output_dir)
        sys.exit(2)

    moved_legacy = _normalize_model_artifacts(output_dir=args.output_dir, model_names=models)
    if moved_legacy > 0:
        print(f"[ARTIFACT] Da sap xep {moved_legacy} file legacy vao thu muc models/AI/<model>.")

    moved_eval_legacy = _normalize_evaluation_artifacts(output_dir=args.output_dir)
    if moved_eval_legacy > 0:
        print(f"[ARTIFACT] Da sap xep {moved_eval_legacy} file evaluation vao models/results/evaluation.")
    
    # Evaluate
    eval_success = True
    if not args.skip_eval:
        eval_success = evaluate_models(
            data_dir=str(data_dir),
            models=models,
            device=args.device,
            output_dir=args.output_dir,
            report_dir=args.report_dir,
            eval_batch_size=int(args.eval_batch_size),
            eval_num_workers=int(args.eval_num_workers),
            max_eval_samples=int(args.max_eval_samples),
        )

    if not eval_success:
        print("[ERROR] Evaluation that bai. Kiem tra log ben tren de sua loi.")
        sys.exit(1)

    # Export training/evaluation summary table after all runs.
    export_training_tables(models=models, output_dir=args.output_dir)
    
    # Display
    display_results(args.output_dir, args.report_dir)
    
    print(f"\n{'='*70}")
    print("✓ DONE!")
    print(f"{'='*70}\n")

    mirrored = mirror_artifacts_to_runtime_dirs(
        [
            args.output_dir,
            args.report_dir,
            str(_training_results_dir(args.output_dir)),
            str(_evaluation_results_dir(args.output_dir)),
        ],
        log_dir=args.log_dir,
        json_dir=args.json_dir,
    )
    print(f"[ARTIFACT] Mirrored logs={mirrored['copied_log']}, json={mirrored['copied_json']}")


if __name__ == "__main__":
    main()
