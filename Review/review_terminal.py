from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.features import PillImageDataset, build_transforms
from src.models.model_factory import load_checkpoint, load_checkpoint_class_to_idx
from src.training.train import train as train_one_model

DEFAULT_MODELS = ["resnet50", "efficientnet_b0", "vit_b_16"]


def discover_data_dir(preferred: str | None) -> Path:
    if preferred:
        root = Path(preferred)
        if not (root / "test").exists():
            raise FileNotFoundError(f"Thiếu thư mục test: {root / 'test'}")
        return root

    for candidate in [Path("data_aligned"), Path("data")]:
        if (candidate / "test").exists():
            return candidate

    raise FileNotFoundError("Không tìm thấy thư mục dữ liệu. Hãy truyền --data-dir.")


def resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        print("[CANH_BAO] CUDA không khả dụng, tự chuyển sang CPU", flush=True)
        return torch.device("cpu")
    return torch.device(device)


def load_weight_from_metrics(models_dir: Path, model_name: str) -> float:
    metrics_path = models_dir / f"{model_name}_epillid_best.metrics.json"
    if not metrics_path.exists():
        return 1.0
    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        w = float(obj.get("best_val_acc", 1.0))
        return w if w > 0 else 1.0
    except Exception:
        return 1.0


def evaluate_single_model(
    model_name: str,
    checkpoint_path: Path,
    loader: DataLoader,
    dataset_class_to_idx: Dict[str, int],
    device: torch.device,
) -> Tuple[Dict[str, float], List[str], List[str]]:
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
        pbar = tqdm(loader, desc=f"Đánh giá {model_name}", leave=False)
        for images, labels, _paths in pbar:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            for true_idx, pred_idx in zip(labels.tolist(), preds.tolist()):
                y_true.append(inv_true.get(int(true_idx), f"class_{int(true_idx)}"))
                y_pred.append(inv_pred.get(int(pred_idx), f"class_{int(pred_idx)}"))

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    row = {
        "model": model_name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "num_samples": int(len(y_true)),
    }
    return row, y_true, y_pred


def evaluate_weighted_ensemble(
    model_names: Sequence[str],
    models_dir: Path,
    loader: DataLoader,
    dataset_class_to_idx: Dict[str, int],
    device: torch.device,
) -> Dict[str, float]:
    inv_true = {v: k for k, v in dataset_class_to_idx.items()}

    loaded = []
    for model_name in model_names:
        ckpt = models_dir / f"{model_name}_epillid_best.pt"
        if not ckpt.exists():
            continue

        ckpt_class_to_idx = load_checkpoint_class_to_idx(str(ckpt), map_location=device)
        num_classes = len(ckpt_class_to_idx) if ckpt_class_to_idx else len(dataset_class_to_idx)
        model = load_checkpoint(
            model_name=model_name,
            num_classes=num_classes,
            checkpoint_path=str(ckpt),
            map_location=device,
        ).to(device)
        model.eval()

        inv_pred = {v: k for k, v in (ckpt_class_to_idx or dataset_class_to_idx).items()}
        weight = load_weight_from_metrics(models_dir, model_name)
        loaded.append((model, inv_pred, weight))

    if not loaded:
        raise RuntimeError("Không tìm thấy checkpoint để xây dựng ensemble")

    y_true: List[str] = []
    y_pred: List[str] = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Đánh giá ensemble_weighted", leave=False)
        for images, labels, _paths in pbar:
            images = images.to(device)
            labels = labels.to(device)
            votes = [dict() for _ in range(images.size(0))]

            for model, inv_pred, weight in loaded:
                probs = F.softmax(model(images), dim=1)
                for i in range(probs.size(0)):
                    for cls_idx, p in enumerate(probs[i].tolist()):
                        cls_name = inv_pred.get(int(cls_idx), f"class_{int(cls_idx)}")
                        votes[i][cls_name] = votes[i].get(cls_name, 0.0) + weight * float(p)

            for i, true_idx in enumerate(labels.tolist()):
                pred_name = max(votes[i].items(), key=lambda x: x[1])[0]
                y_true.append(inv_true.get(int(true_idx), f"class_{int(true_idx)}"))
                y_pred.append(pred_name)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    return {
        "model": "ensemble_weighted",
        "accuracy": acc,
        "macro_f1": macro_f1,
        "num_samples": int(len(y_true)),
    }


def verdict(acc: float, macro_f1: float) -> str:
    score = 0.6 * acc + 0.4 * macro_f1
    if score >= 0.8:
        return "rat_tot"
    if score >= 0.6:
        return "tot"
    if score >= 0.4:
        return "trung_binh"
    return "kem"


def blended_score(acc: float, macro_f1: float) -> float:
    return 0.6 * acc + 0.4 * macro_f1


def print_table(rows: List[Dict[str, float]]) -> None:
    headers = ["model", "accuracy", "macro_f1", "blend", "num_samples", "verdict"]

    view_rows = []
    for r in rows:
        view_rows.append(
            {
                "model": str(r["model"]),
                "accuracy": f"{float(r['accuracy']):.4f}",
                "macro_f1": f"{float(r['macro_f1']):.4f}",
                "blend": f"{blended_score(float(r['accuracy']), float(r['macro_f1'])):.4f}",
                "num_samples": str(int(float(r["num_samples"]))),
                "verdict": verdict(float(r["accuracy"]), float(r["macro_f1"])),
            }
        )

    if not view_rows:
        col_width = {h: len(h) for h in headers}
    else:
        col_width = {
            h: max(len(h), max((len(v[h]) for v in view_rows), default=0)) for h in headers
        }

    line = "+" + "+".join("-" * (col_width[h] + 2) for h in headers) + "+"
    print(line)
    print("| " + " | ".join(h.ljust(col_width[h]) for h in headers) + " |")
    print(line)
    for row in view_rows:
        print("| " + " | ".join(row[h].ljust(col_width[h]) for h in headers) + " |")
    print(line)


def load_history(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    return []


def save_history(path: Path, entries: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def _history_signature(entry: Dict[str, object]) -> str:
    payload = {
        "session_id": entry.get("session_id"),
        "round": entry.get("round"),
        "train_before_review": entry.get("train_before_review"),
        "lr": entry.get("lr"),
        "weight_decay": entry.get("weight_decay"),
        "epochs": entry.get("epochs"),
        "best_model": entry.get("best_model"),
        "best_accuracy": entry.get("best_accuracy"),
        "best_macro_f1": entry.get("best_macro_f1"),
        "rows": entry.get("rows"),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def dedupe_history(entries: List[Dict[str, object]]) -> List[Dict[str, object]]:
    seen: set[str] = set()
    deduped: List[Dict[str, object]] = []
    for entry in entries:
        sig = _history_signature(entry)
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(entry)
    return deduped


def propose_next_hparams(
    history: List[Dict[str, object]],
    current_lr: float,
    current_weight_decay: float,
    current_epochs: int,
) -> Tuple[float, float, int]:
    if len(history) < 2:
        return current_lr, current_weight_decay, current_epochs

    prev_score = float(history[-2].get("best_blend", 0.0))
    last_score = float(history[-1].get("best_blend", 0.0))

    # Nếu vòng gần nhất tốt lên, fine-tune nhẹ để ổn định hơn.
    if last_score >= prev_score:
        next_lr = max(1e-5, current_lr * 0.9)
        next_wd = min(5e-4, max(1e-6, current_weight_decay))
        next_epochs = min(30, current_epochs + 1)
        return next_lr, next_wd, next_epochs

    # Nếu giảm chất lượng, thử giảm LR mạnh hơn và tăng epochs.
    next_lr = max(1e-5, current_lr * 0.7)
    next_wd = min(1e-3, max(1e-6, current_weight_decay * 1.2))
    next_epochs = min(40, current_epochs + 2)
    return next_lr, next_wd, next_epochs


def train_all_models(
    model_names: Sequence[str],
    data_dir: Path,
    models_dir: Path,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    early_stop_patience: int,
    pretrained: bool,
) -> None:
    for idx, model_name in enumerate(model_names, start=1):
        print(
            f"[RT][TRAIN] {idx}/{len(model_names)} model={model_name} "
            f"epochs={epochs} lr={lr:.6f} wd={weight_decay:.6f}",
            flush=True,
        )
        train_args = argparse.Namespace(
            data_dir=str(data_dir),
            model=model_name,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            num_workers=num_workers,
            device=device,
            output_dir=str(models_dir),
            early_stop_patience=early_stop_patience,
            save_curves=False,
            pretrained=pretrained,
        )
        train_one_model(train_args)


def cleanup_old_artifacts(
    models_dir: Path,
    model_names: Sequence[str],
    keep_report_dirs: int,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Xoa file cu khong can thiet de giam dung luong.

    Nguyen tac an toan:
    - Chi xoa file .pt/.json o muc goc models khong nam trong danh sach can giu.
    - Trong models/reports, chi giu N thu muc moi nhat, xoa cac thu muc cu hon.
    """
    keep_names = {"terminal_review_history.json"}
    for model_name in model_names:
        keep_names.add(f"{model_name}_epillid_best.pt")
        keep_names.add(f"{model_name}_epillid_best.metrics.json")
        keep_names.add(f"{model_name}_epillid_history.json")

    removed_pt = 0
    removed_json = 0
    removed_report_dirs = 0

    # Don dep .pt/.json o muc goc models
    for path in models_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".pt", ".json"}:
            continue
        if path.name in keep_names:
            continue

        if not dry_run:
            try:
                path.unlink()
            except Exception:
                continue

        if path.suffix.lower() == ".pt":
            removed_pt += 1
        else:
            removed_json += 1

    # Don dep reports cu
    reports_dir = models_dir / "reports"
    if reports_dir.exists() and reports_dir.is_dir():
        dirs = [d for d in reports_dir.iterdir() if d.is_dir()]
        dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        stale_dirs = dirs[max(0, keep_report_dirs):]
        for stale in stale_dirs:
            if not dry_run:
                try:
                    shutil.rmtree(stale)
                except Exception:
                    continue
            removed_report_dirs += 1

    return {
        "removed_pt": removed_pt,
        "removed_json": removed_json,
        "removed_report_dirs": removed_report_dirs,
    }


def parse_args() -> argparse.Namespace:
    formatter = lambda prog: argparse.HelpFormatter(prog, width=120)
    parser = argparse.ArgumentParser(
        description="Tự review mô hình: in bảng phân tích trực tiếp ra terminal (không tạo ảnh)",
        formatter_class=formatter,
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument(
        "--model-list",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Danh sách mô hình phân tách bằng dấu phẩy",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=1, help="Số vòng train + review")
    parser.add_argument(
        "--train-before-review",
        action="store_true",
        default=False,
        help="Train toàn bộ model trước mỗi vòng review",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dùng trọng số pretrained khi train",
    )
    parser.add_argument(
        "--history-file",
        type=str,
        default="models/terminal_review_history.json",
        help="Đường dẫn file lịch sử tối ưu hóa",
    )
    parser.add_argument(
        "--with-ensemble",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bao gồm thêm dòng ensemble có trọng số",
    )
    parser.add_argument(
        "--auto-cleanup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tu dong xoa file .pt/.json cu khong can thiet sau moi vong",
    )
    parser.add_argument(
        "--keep-report-dirs",
        type=int,
        default=3,
        help="So thu muc report moi nhat se duoc giu lai",
    )
    parser.add_argument(
        "--cleanup-dry-run",
        action="store_true",
        default=False,
        help="Chi in ket qua don dep, khong xoa that",
    )
    return parser.parse_args()


def main_with_args(args: argparse.Namespace) -> None:

    data_root = discover_data_dir(args.data_dir)
    test_root = data_root / "test"
    models_dir = Path(args.models_dir)
    device = resolve_device(args.device)
    history_path = Path(args.history_file)

    print(f"[THONG_TIN] thu_muc_du_lieu={data_root}", flush=True)
    print(f"[THONG_TIN] thu_muc_test={test_root}", flush=True)
    print(f"[THONG_TIN] thu_muc_model={models_dir}", flush=True)
    print(f"[THONG_TIN] thiet_bi={device}", flush=True)
    print(f"[THONG_TIN] so_vong={args.rounds}, train_truoc_review={args.train_before_review}", flush=True)
    print(f"[THONG_TIN] file_lich_su={history_path}", flush=True)
    print(
        f"[THONG_TIN] auto_cleanup={args.auto_cleanup}, keep_report_dirs={args.keep_report_dirs}, dry_run={args.cleanup_dry_run}",
        flush=True,
    )

    dataset = PillImageDataset(str(test_root), transform=build_transforms(train=False))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model_names = [m.strip() for m in args.model_list.split(",") if m.strip()]

    history = load_history(history_path)
    session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    cur_lr = float(args.lr)
    cur_wd = float(args.weight_decay)
    cur_epochs = int(args.epochs)

    final_rows: List[Dict[str, float]] = []
    for round_idx in range(1, max(1, int(args.rounds)) + 1):
        print(
            f"\n[THOI_GIAN_THUC][VONG {round_idx}/{max(1, int(args.rounds))}] "
            f"bat_dau lr={cur_lr:.6f}, wd={cur_wd:.6f}, epochs={cur_epochs}",
            flush=True,
        )

        if args.train_before_review:
            train_all_models(
                model_names=model_names,
                data_dir=data_root,
                models_dir=models_dir,
                device=str(device),
                epochs=cur_epochs,
                batch_size=args.batch_size,
                lr=cur_lr,
                weight_decay=cur_wd,
                num_workers=args.num_workers,
                early_stop_patience=args.early_stop_patience,
                pretrained=args.pretrained,
            )

        rows: List[Dict[str, float]] = []
        for model_name in model_names:
            ckpt = models_dir / f"{model_name}_epillid_best.pt"
            if not ckpt.exists():
                print(f"[CANH_BAO] Bo qua {model_name}: thiếu checkpoint {ckpt}", flush=True)
                continue

            row, _y_true, _y_pred = evaluate_single_model(
                model_name=model_name,
                checkpoint_path=ckpt,
                loader=loader,
                dataset_class_to_idx=dataset.class_to_idx,
                device=device,
            )
            rows.append(row)
            print(
                f"[XONG] {model_name}: accuracy={row['accuracy']:.4f}, macro_f1={row['macro_f1']:.4f}",
                flush=True,
            )

        if not rows:
            raise RuntimeError("Không có checkpoint hợp lệ để đánh giá")

        if args.with_ensemble:
            ens_row = evaluate_weighted_ensemble(
                model_names=model_names,
                models_dir=models_dir,
                loader=loader,
                dataset_class_to_idx=dataset.class_to_idx,
                device=device,
            )
            rows.append(ens_row)
            print(
                f"[XONG] ensemble_weighted: accuracy={ens_row['accuracy']:.4f}, macro_f1={ens_row['macro_f1']:.4f}",
                flush=True,
            )

        rows = sorted(
            rows,
            key=lambda r: (blended_score(float(r["accuracy"]), float(r["macro_f1"]))),
            reverse=True,
        )
        final_rows = rows

        print("\n=== BANG TU REVIEW (CHI HIEN THI TERMINAL) ===")
        print_table(rows)

        best = rows[0]
        best_blend = blended_score(float(best["accuracy"]), float(best["macro_f1"]))
        print(
            f"\n[TONG_KET][VONG {round_idx}] model_tot_nhat={best['model']}, "
            f"accuracy={best['accuracy']:.4f}, macro_f1={best['macro_f1']:.4f}, blend={best_blend:.4f}",
            flush=True,
        )

        history.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "session_id": session_id,
                "round": round_idx,
                "train_before_review": bool(args.train_before_review),
                "lr": cur_lr,
                "weight_decay": cur_wd,
                "epochs": max(1, cur_epochs) if bool(args.train_before_review) else cur_epochs,
                "best_model": best["model"],
                "best_accuracy": float(best["accuracy"]),
                "best_macro_f1": float(best["macro_f1"]),
                "best_blend": best_blend,
                "rows": rows,
            }
        )
        history = dedupe_history(history)
        save_history(history_path, history)

        cur_lr, cur_wd, cur_epochs = propose_next_hparams(
            history=history,
            current_lr=cur_lr,
            current_weight_decay=cur_wd,
            current_epochs=cur_epochs,
        )

        print(
            f"[THOI_GIAN_THUC][GOI_Y_VONG_SAU] lr={cur_lr:.6f}, wd={cur_wd:.6f}, epochs={cur_epochs}",
            flush=True,
        )

        if args.auto_cleanup:
            cleanup_info = cleanup_old_artifacts(
                models_dir=models_dir,
                model_names=model_names,
                keep_report_dirs=max(0, int(args.keep_report_dirs)),
                dry_run=bool(args.cleanup_dry_run),
            )
            print(
                "[DON_DEP] "
                f"xoa_pt={cleanup_info['removed_pt']}, "
                f"xoa_json={cleanup_info['removed_json']}, "
                f"xoa_thu_muc_report={cleanup_info['removed_report_dirs']}",
                flush=True,
            )

    if final_rows:
        best_final = final_rows[0]
        print(
            f"\n[KET_THUC] model_tot_nhat={best_final['model']} "
            f"accuracy={best_final['accuracy']:.4f} macro_f1={best_final['macro_f1']:.4f}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    main_with_args(args)


if __name__ == "__main__":
    main()
