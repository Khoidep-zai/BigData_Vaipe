"""Cau hinh toi uu: tang regularization, giu train/val gap nho, hoi tu nhanh."""

# Muc tieu: train/val chay song song, accuracy cao, thoi gian train giam.
# Thay doi so voi phien ban cu:
#   - Tang label_smoothing, mixup_alpha, weight_decay → giam overfitting
#   - Giam lr → on dinh hoi tu
#   - Giam epochs (28/32) + patience (6/7) → dung som khi da tot
#   - Tang freeze_backbone_epochs (3-5) → head on dinh truoc khi unfreeze
#   - Giam backbone_lr_scale (0.15) → backbone hoc cham hon head
OPTIMAL_CONFIGS = {
    "resnet50": {
        "lr": 6e-5,
        "weight_decay": 1.2e-3,
        "label_smoothing": 0.16,
        "mixup_alpha": 0.35,
        "epochs": 28,
        "batch_size": 16,
        "early_stop_patience": 6,
        "max_train_val_gap": 0.15,
        "freeze_backbone_epochs": 3,
        "backbone_lr_scale": 0.15,
        "ema_decay": 0.998,
        "tta_views": 3,
    },
    "efficientnet_b0": {
        "lr": 7e-5,
        "weight_decay": 1e-3,
        "label_smoothing": 0.15,
        "mixup_alpha": 0.33,
        "epochs": 28,
        "batch_size": 16,
        "early_stop_patience": 6,
        "max_train_val_gap": 0.15,
        "freeze_backbone_epochs": 3,
        "backbone_lr_scale": 0.15,
        "ema_decay": 0.998,
        "tta_views": 3,
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
        "backbone_lr_scale": 0.12,
        "ema_decay": 0.998,
        "tta_views": 5,
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
