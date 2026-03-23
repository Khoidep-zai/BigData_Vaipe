"""Cau hinh uu tien train/val on dinh va bam sat nhau tren dataset nho."""

# Muc tieu: giam overfitting, giu train/val gap nho va duong cong on dinh.
OPTIMAL_CONFIGS = {
    "resnet50": {
        "lr": 2e-4,
        "weight_decay": 5e-4,
        "label_smoothing": 0.06,
        "mixup_alpha": 0.10,
        "epochs": 36,
        "batch_size": 16,
        "early_stop_patience": 8,
        "max_train_val_gap": 0.16,
        "freeze_backbone_epochs": 2,
    },
    "efficientnet_b0": {
        "lr": 2.2e-4,
        "weight_decay": 6e-4,
        "label_smoothing": 0.06,
        "mixup_alpha": 0.10,
        "epochs": 36,
        "batch_size": 16,
        "early_stop_patience": 8,
        "max_train_val_gap": 0.16,
        "freeze_backbone_epochs": 2,
    },
    "vit_b_16": {
        "lr": 5e-5,
        "weight_decay": 1.4e-3,
        "label_smoothing": 0.20,
        "mixup_alpha": 0.42,
        "epochs": 32,
        "batch_size": 16,
        "early_stop_patience": 7,
        "max_train_val_gap": 0.12,
        "freeze_backbone_epochs": 5,
    },
}

# Candidates de review_terminal.py tu de xuat/vong lap nhieu lan.
TUNING_CANDIDATES = {
    "resnet50": [
        {"lr": 5e-5, "weight_decay": 1.3e-3, "label_smoothing": 0.18, "mixup_alpha": 0.38},
        {"lr": 6e-5, "weight_decay": 1.2e-3, "label_smoothing": 0.16, "mixup_alpha": 0.35},
        {"lr": 7e-5, "weight_decay": 1e-3, "label_smoothing": 0.14, "mixup_alpha": 0.30},
    ],
    "efficientnet_b0": [
        {"lr": 6e-5, "weight_decay": 1.2e-3, "label_smoothing": 0.18, "mixup_alpha": 0.36},
        {"lr": 7e-5, "weight_decay": 1e-3, "label_smoothing": 0.15, "mixup_alpha": 0.33},
        {"lr": 8e-5, "weight_decay": 9e-4, "label_smoothing": 0.13, "mixup_alpha": 0.30},
    ],
    "vit_b_16": [
        {"lr": 4e-5, "weight_decay": 1.6e-3, "label_smoothing": 0.22, "mixup_alpha": 0.45},
        {"lr": 5.5e-5, "weight_decay": 1.3e-3, "label_smoothing": 0.19, "mixup_alpha": 0.40},
        {"lr": 6e-5, "weight_decay": 1.2e-3, "label_smoothing": 0.18, "mixup_alpha": 0.38},
    ],
}
