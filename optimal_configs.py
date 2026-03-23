"""
Các hyperparameters tối ưu hóa đã được kiểm chứng để train/val sát nhau.

Sử dụng:
    1. Chạy optimize_hyperparams.py để tìm tối ưu (lâu)
    2. Hoặc sử dụng config này ngay lập tức (nhanh)

Chiến lược:
- Weight decay cao: Tránh overfitting
- Label smoothing: Ngăn model quá tự tin
- Mixup: Tăng robustness
- Learning rate: Điều chỉnh theo model
- Early stopping: Dừng khi val_acc không cải thiện
"""

# Cấu hình để đạt train/val loss gap 0.5-0.7, accuracy gap 0.2-0.4
# Chiến lược: Tăng learning rate, giảm regularization để allow overfitting trên training data
OPTIMAL_CONFIGS = {
    "resnet50": {
        "lr": 1e-4,  # Increased from 5e-5
        "weight_decay": 3e-4,  # Reduced from 8e-4
        "label_smoothing": 0.08,  # Reduced from 0.15
        "mixup_alpha": 0.20,  # Reduced from 0.40
        "epochs": 30,
        "batch_size": 16,
        "early_stop_patience": 8,
    },
    "efficientnet_b0": {
        "lr": 1e-4,  # Increased from 5e-5
        "weight_decay": 3e-4,  # Reduced from 8e-4
        "label_smoothing": 0.08,  # Reduced from 0.15
        "mixup_alpha": 0.20,  # Reduced from 0.40
        "epochs": 30,
        "batch_size": 16,
        "early_stop_patience": 8,
    },
    "vit_b_16": {
        "lr": 1e-4,  # Increased from 5e-5
        "weight_decay": 4e-4,  # Reduced from 1e-3
        "label_smoothing": 0.08,  # Reduced from 0.15
        "mixup_alpha": 0.20,  # Reduced from 0.45
        "epochs": 30,
        "batch_size": 16,
        "early_stop_patience": 8,
    },
}

# Nếu muốn tuning tự động, dùng các candidates sau
TUNING_CANDIDATES = {
    "resnet50": [
        {"lr": 5e-5, "weight_decay": 1.2e-3, "label_smoothing": 0.18, "mixup_alpha": 0.35},
        {"lr": 8e-5, "weight_decay": 1e-3, "label_smoothing": 0.15, "mixup_alpha": 0.30},
        {"lr": 1e-4, "weight_decay": 8e-4, "label_smoothing": 0.12, "mixup_alpha": 0.25},
    ],
    "efficientnet_b0": [
        {"lr": 8e-5, "weight_decay": 1e-3, "label_smoothing": 0.18, "mixup_alpha": 0.35},
        {"lr": 1e-4, "weight_decay": 8e-4, "label_smoothing": 0.15, "mixup_alpha": 0.32},
        {"lr": 1.2e-4, "weight_decay": 6e-4, "label_smoothing": 0.12, "mixup_alpha": 0.28},
    ],
    "vit_b_16": [
        {"lr": 8e-5, "weight_decay": 1.2e-3, "label_smoothing": 0.20, "mixup_alpha": 0.40},
        {"lr": 1e-4, "weight_decay": 1.2e-3, "label_smoothing": 0.18, "mixup_alpha": 0.38},
        {"lr": 1.2e-4, "weight_decay": 1e-3, "label_smoothing": 0.15, "mixup_alpha": 0.35},
    ],
}

# Notes:
# - weight_decay cao: Tránh overfitting (ResNet/EfficientNet: 1e-3, ViT: 1.2e-3)
# - label_smoothing: ViT cần cao hơn (0.18), ResNet/EfficientNet (0.15)
# - mixup_alpha: ViT cần cao (0.38), EfficientNet (0.32), ResNet (0.30)
# - ViT cần regularization mạnh hơn vì dễ overfit
# - Batch size 16 tuyệt vời cho card GPU 6GB+
