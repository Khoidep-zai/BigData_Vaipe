from __future__ import annotations

import argparse

from src import train as train_module
from src import pipeline as pipeline_module
import review_terminal as review_module


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLI for training pill models (single model or full pipeline)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "single", "optimize"],
        help="all: full pipeline, single: one model train, optimize: realtime train+review rounds",
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
    parser.add_argument("--report-dir", type=str, default="models/reports/latest")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--keep-report-dirs", type=int, default=3)
    parser.add_argument("--cleanup-dry-run", action="store_true", default=False)

    args, _ = parser.parse_known_args()

    if args.mode == "all":
        pipeline_args = argparse.Namespace(
            data_dir=args.data_dir,
            models="resnet50,efficientnet_b0,vit_b_16",
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            mixup_alpha=args.mixup_alpha,
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
        pipeline_module.run_pipeline(pipeline_args)
        return

    if args.mode == "optimize":
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
            early_stop_patience=args.early_stop_patience,
            max_train_val_gap=args.max_train_val_gap,
            seed=args.seed,
            rounds=args.rounds,
            train_before_review=True,
            pretrained=True,
            history_file="models/terminal_review_history.json",
            with_ensemble=True,
            auto_cleanup=True,
            keep_report_dirs=args.keep_report_dirs,
            cleanup_dry_run=args.cleanup_dry_run,
        )
        review_module.main_with_args(review_args)
        return

    # single mode
    train_args = argparse.Namespace(
        data_dir=args.data_dir or "data_aligned",
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        num_workers=args.num_workers,
        device=args.device,
        output_dir=args.output_dir,
        early_stop_patience=args.early_stop_patience,
        max_train_val_gap=args.max_train_val_gap,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        seed=args.seed,
        save_curves=True,
        pretrained=True,
    )
    train_module.train(train_args)


if __name__ == "__main__":
    main()

