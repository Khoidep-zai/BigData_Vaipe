from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.data_setup import discover_or_prepare_data_dir, prepare_metadata_artifacts
from src.inference.prescription_matching import (
    format_pretty_summary,
    match_pills_to_prescription,
    result_to_dict,
    write_result_csv,
)
from src.orchestration.pipeline import run_pipeline
from src.training.train import train as train_one_model
from src.utils.model_paths import single_mode_output_dir
from src.utils.runtime_artifacts import ensure_runtime_dirs, mirror_artifacts_to_runtime_dirs
from Review import review_terminal as review_module


def main() -> None:
    # Điểm truy cập dòng lệnh (CLI) duy nhất để chuyển đổi giữa các chế độ: chạy toàn bộ quy trình, huấn luyện 1 mô hình, hoặc chạy vòng lặp tối ưu hóa.
    parser = argparse.ArgumentParser(
        description="CLI for training pill models (single model or full pipeline)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "single", "optimize", "prescription_match"],
        help=(
            "all: full pipeline, single: one model train, optimize: realtime train+review rounds, "
            "prescription_match: split pills into in-prescription/out-of-prescription"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet50", "efficientnet_b0", "vit_b_16"],
        help="Backbone name for single mode",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Dataset root containing train/val/test. If omitted in all mode, auto-discovery is used.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Number of training epochs",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument(
        "--augment-profile",
        type=str,
        default="default",
        choices=["default", "lab6_stable"],
        help="Train augmentation profile. 'lab6_stable' applies Lab6-inspired stable transforms.",
    )
    parser.add_argument(
        "--vector-grayscale-prob",
        type=float,
        default=0.0,
        help="Probability of grayscale augmentation during train (Lab6 vector-inspired).",
    )
    parser.add_argument("--backbone-lr-scale", type=float, default=0.2)
    parser.add_argument("--ema-decay", type=float, default=0.997)
    parser.add_argument("--tta-views", type=int, default=3)
    parser.add_argument("--train-metric-every", type=int, default=3)
    parser.add_argument("--quick-val-tta-views", type=int, default=1)
    parser.add_argument(
        "--full-tta-on-save",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
    )
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--max-train-val-gap", type=float, default=0.18)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--report-dir", type=str, default="")
    parser.add_argument("--log-dir", type=str, default="log")
    parser.add_argument("--json-dir", type=str, default="json")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--keep-report-dirs", type=int, default=3)
    parser.add_argument("--cleanup-dry-run", action="store_true", default=False)
    parser.add_argument(
        "--fast-train",
        action="store_true",
        default=False,
        help="Reduce training compute (fewer epochs, no EMA, no val-TTA) to iterate faster.",
    )
    parser.add_argument(
        "--skip-metadata-artifacts",
        action="store_true",
        default=False,
        help="Skip metadata clean/vector export step before training.",
    )
    parser.add_argument(
        "--prescription-image",
        type=str,
        default="",
        help="[prescription_match] One prescription image path.",
    )
    parser.add_argument(
        "--pill-images",
        nargs="*",
        default=[],
        help="[prescription_match] One or more pill image paths.",
    )
    parser.add_argument(
        "--annotations-csv",
        type=str,
        default="data/csv/Prescription_Pill_Annotations.csv",
        help="[prescription_match] Annotation CSV generated by data/Loc_du_lieu.py.",
    )
    parser.add_argument(
        "--prescription-index-csv",
        type=str,
        default="data/csv/Prescription_Image_Index.csv",
        help="[prescription_match] Prescription index CSV generated by data/Loc_du_lieu.py.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default="data/csv/Medicine_Details_Training.csv",
        help="[prescription_match] Metadata CSV used to map class id to medicine name.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="[prescription_match] Optional output JSON path. Print to stdout if omitted.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="[prescription_match] Optional output CSV path (one detection per row).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=False,
        help="[prescription_match] Print concise terminal summary (counts per pill image).",
    )

    args, _ = parser.parse_known_args()
    if not str(args.report_dir).strip():
        args.report_dir = str(Path(args.output_dir) / "reports" / "latest")
    ensure_runtime_dirs(log_dir=args.log_dir, json_dir=args.json_dir)

    if args.mode == "prescription_match":
        if not args.prescription_image:
            raise SystemExit("[ERROR] --prescription-image la bat buoc cho mode prescription_match")
        if not args.pill_images:
            raise SystemExit("[ERROR] --pill-images can it nhat 1 anh cho mode prescription_match")

        result = match_pills_to_prescription(
            prescription_image=args.prescription_image,
            pill_images=args.pill_images,
            annotations_csv=args.annotations_csv,
            prescription_index_csv=args.prescription_index_csv,
            metadata_csv=args.metadata_csv,
        )
        payload = result_to_dict(result)

        output_json_path = Path(args.output_json) if args.output_json else Path(args.json_dir) / "prescription_match_latest.json"
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[OK] Da ghi ket qua: {output_json_path}")
        if not args.output_json:
            # Use ASCII-safe output for Windows consoles with limited codepages.
            print(json.dumps(payload, ensure_ascii=True, indent=2))

        if args.output_csv:
            csv_path = write_result_csv(result, args.output_csv)
            print(f"[OK] Da ghi CSV: {csv_path}")

        if args.pretty:
            print(format_pretty_summary(result))
        return

    try:
        args.data_dir = discover_or_prepare_data_dir(preferred=args.data_dir, seed=int(args.seed))
    except Exception as exc:
        raise SystemExit(f"[ERROR] Khong the chuan bi du lieu train: {exc}")

    if not args.skip_metadata_artifacts:
        metadata_artifacts_obj = prepare_metadata_artifacts(data_root="data")
        if metadata_artifacts_obj:
            metadata_artifacts = metadata_artifacts_obj if isinstance(metadata_artifacts_obj, dict) else {}
            clean_summary = metadata_artifacts.get("clean_summary", {})
            clean_csv = clean_summary.get("output_csv", "") if isinstance(clean_summary, dict) else ""
            vector_csv = metadata_artifacts.get("vector_csv", "")
            print(f"[META] Clean metadata CSV: {clean_csv}")
            print(f"[META] Metadata vector CSV: {vector_csv}")

    if args.fast_train:
        args.epochs = max(6, int(round(float(args.epochs) * 0.45)))
        args.tta_views = 1
        args.ema_decay = 0.0
        args.train_metric_every = max(int(args.train_metric_every), 4)
        args.quick_val_tta_views = 1
        args.full_tta_on_save = True
        args.early_stop_patience = min(int(args.early_stop_patience), 4)
        args.freeze_backbone_epochs = min(int(args.freeze_backbone_epochs), 1)
        print(
            "[FAST] Enabled: "
            f"epochs={args.epochs}, tta_views={args.tta_views}, ema_decay={args.ema_decay}, "
            f"train_metric_every={args.train_metric_every}, patience={args.early_stop_patience}"
        )

    if args.mode == "all":
        # Chuyển toàn bộ tham số dòng lệnh sang tầng điều phối (orchestration) để thực hiện: huấn luyện + đánh giá + báo cáo.
        pipeline_args = argparse.Namespace(
            data_dir=args.data_dir,
            models="resnet50,efficientnet_b0,vit_b_16",
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            mixup_alpha=args.mixup_alpha,
            augment_profile=args.augment_profile,
            vector_grayscale_prob=args.vector_grayscale_prob,
            backbone_lr_scale=args.backbone_lr_scale,
            ema_decay=args.ema_decay,
            tta_views=args.tta_views,
            train_metric_every=args.train_metric_every,
            quick_val_tta_views=args.quick_val_tta_views,
            full_tta_on_save=args.full_tta_on_save,
            deterministic=args.deterministic,
            num_workers=args.num_workers,
            device=args.device,
            early_stop_patience=args.early_stop_patience,
            max_train_val_gap=args.max_train_val_gap,
            seed=args.seed,
            output_dir=args.output_dir,
            report_dir=args.report_dir,
            save_curves=True,
            pretrained=True,
        )
        run_pipeline(pipeline_args)
        mirrored = mirror_artifacts_to_runtime_dirs(
            [args.output_dir, args.report_dir],
            log_dir=args.log_dir,
            json_dir=args.json_dir,
        )
        print(f"[ARTIFACT] Mirrored logs={mirrored['copied_log']}, json={mirrored['copied_json']}")
        return

    if args.mode == "optimize":
        # Chạy các vòng lặp huấn luyện-đánh giá ngay trên cửa sổ lệnh (terminal) và lưu lại lịch sử quá trình tối ưu.
        review_args = argparse.Namespace(
            data_dir=args.data_dir,
            models_dir=args.output_dir,
            model_list="resnet50,efficientnet_b0,vit_b_16",
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            mixup_alpha=args.mixup_alpha,
            augment_profile=args.augment_profile,
            vector_grayscale_prob=args.vector_grayscale_prob,
            backbone_lr_scale=args.backbone_lr_scale,
            ema_decay=args.ema_decay,
            tta_views=args.tta_views,
            train_metric_every=args.train_metric_every,
            quick_val_tta_views=args.quick_val_tta_views,
            full_tta_on_save=args.full_tta_on_save,
            deterministic=args.deterministic,
            early_stop_patience=args.early_stop_patience,
            max_train_val_gap=args.max_train_val_gap,
            seed=args.seed,
            rounds=args.rounds,
            train_before_review=True,
            pretrained=True,
            history_file=str(Path(args.json_dir) / "terminal_review_history.json"),
            with_ensemble=True,
            auto_cleanup=True,
            keep_report_dirs=args.keep_report_dirs,
            cleanup_dry_run=args.cleanup_dry_run,
        )
        review_module.main_with_args(review_args)
        mirrored = mirror_artifacts_to_runtime_dirs(
            [args.output_dir, args.report_dir],
            log_dir=args.log_dir,
            json_dir=args.json_dir,
        )
        print(f"[ARTIFACT] Mirrored logs={mirrored['copied_log']}, json={mirrored['copied_json']}")
        return

    # Chế độ single: lệnh chạy nhanh để huấn luyện duy nhất một loại mô hình (backbone).
    single_output_dir = single_mode_output_dir(args.output_dir, args.model)
    train_args = argparse.Namespace(
        data_dir=args.data_dir,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        augment_profile=args.augment_profile,
        vector_grayscale_prob=args.vector_grayscale_prob,
        backbone_lr_scale=args.backbone_lr_scale,
        ema_decay=args.ema_decay,
        tta_views=args.tta_views,
        train_metric_every=args.train_metric_every,
        quick_val_tta_views=args.quick_val_tta_views,
        full_tta_on_save=args.full_tta_on_save,
        deterministic=args.deterministic,
        num_workers=args.num_workers,
        device=args.device,
        output_dir=str(single_output_dir),
        early_stop_patience=args.early_stop_patience,
        max_train_val_gap=args.max_train_val_gap,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        seed=args.seed,
        save_curves=True,
        pretrained=True,
    )
    train_one_model(train_args)
    mirrored = mirror_artifacts_to_runtime_dirs(
        [args.output_dir, args.report_dir],
        log_dir=args.log_dir,
        json_dir=args.json_dir,
    )
    print(f"[ARTIFACT] Mirrored logs={mirrored['copied_log']}, json={mirrored['copied_json']}")


if __name__ == "__main__":
    main()

