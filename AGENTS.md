# AGENTS.md — THUOC Project
> Đọc file này TRƯỚC KHI viết bất kỳ dòng code nào.
> Compatible: Claude Code · Cursor · Windsurf · GitHub Copilot · Aider · Gemini CLI

---

## ROLE

Bạn là ML engineer đang phát triển **THUOC** — hệ thống phân loại viên thuốc từ ảnh bằng deep learning.

```
Task      : Image classification · 8 classes · pill images
Models    : ResNet50 · EfficientNet-B0 · ViT-B/16 (ensemble)
Framework : PyTorch + torchvision
Python    : 3.10+
Hardware  : CUDA GPU (auto-fallback CPU)
```

---

## COMMANDS — CHẠY TRƯỚC KHI LÀM BẤT CỨ ĐIỀU GÌ

```bash
# Cài thư viện
pip install -r requirements.txt

# Full pipeline: train → evaluate → ensemble → report  [LỆNH CHÍNH]
python run_all.py

# Train 1 model đơn lẻ
python train_cli.py --mode single --model resnet50 --epochs 28
python train_cli.py --mode single --model efficientnet_b0 --epochs 28
python train_cli.py --mode single --model vit_b_16 --epochs 32

# Hyperparameter tuning nhiều vòng (train → review → đề xuất config → lặp)
python train_cli.py --mode optimize --rounds 3 --epochs 12

# Chỉ evaluate checkpoint đã có (không train lại)
python run_all.py --compare-only

# Dùng CPU (không có GPU)
python run_all.py --device cpu

# Smoke test nhanh (2 epoch — dùng khi debug, không dùng để đánh giá)
python train_cli.py --mode single --model resnet50 --epochs 2 --batch-size 4

# Unit tests
python -m pytest tests/test_features.py tests/test_inference_utils.py tests/test_metadata.py -q
```

---

## KIẾN TRÚC HỆ THỐNG

### Pipeline 7 tầng

```
[1] DATA           data_aligned/ (ưu tiên) hoặc data/
                   Bắt buộc: train/ · val/ · test/
                   Cấu trúc: split/class/*.jpg
        │
        ▼
[2] DATA MODULE    PillImageDataset → transforms → DataLoader
                   Train: augmentation ON   |   Val/Test: deterministic
        │
        ▼
[3] TRAINING       ResNet50 · EfficientNet-B0 · ViT-B/16
                   Stage 1: freeze backbone → Stage 2: unfreeze full model
                   Kỹ thuật: Label Smoothing · Mixup · Grad Clip · EMA · TTA · Early Stop
        │
        ▼
[4] CHECKPOINT     <model>_epillid_best.pt  (lưu khi val_acc tốt nhất)
        │
        ▼
[5] EVALUATION     accuracy · macro-F1 · confusion matrix / model
        │
        ▼
[6] ENSEMBLE       Weighted soft-voting từ xác suất 3 model
                   P_ens(c) = Σ(w_m × P_m(c)) / Σ(w_m)
        │
        ▼
[7] OUTPUT         models/ → checkpoints · metrics · history · plots · reports
```

### Module boundaries

| Module | Trách nhiệm | KHÔNG được làm |
|--------|------------|----------------|
| `src/data/` | Chuẩn hóa data, transform, metadata | Chứa logic train/eval |
| `src/models/` | Tạo backbone, load checkpoint | Chứa train loop |
| `src/training/` | Train loop, val loop, save best | Gọi trực tiếp DB/API ngoài |
| `src/orchestration/` | Điều phối toàn bộ pipeline | Chứa model logic |
| `src/evaluation/` | Tính metrics, tạo report | Thay đổi model weights |
| `src/inference/` | Predict 1 ảnh, so sánh 2 ảnh | Chứa train logic |
| `src/learning/` | Ghi feedback, hard examples | Tự sửa model |
| `Review/` | Vòng lặp review + đề xuất HP | Chứa source model |

---

## CẤU TRÚC THƯ MỤC — VAI TRÒ TỪNG FILE

```
THUOC/
├── run_all.py                         # Entrypoint chính — chạy cái này trước
├── train_cli.py                       # CLI: all / single / optimize
├── THUOC_Colab_Train_Evaluate.ipynb   # Colab workflow
├── requirements.txt
├── AGENTS.md                          # File này
├── README.md
│
├── src/
│   ├── data/
│   │   ├── features.py                # PillImageDataset · transforms · class_to_idx
│   │   ├── metadata.py               # Parse CSV · tokenize · map class→metadata
│   │   └── build_epillid_data.py     # Align raw split → cấu trúc train/val/test
│   ├── models/
│   │   ├── resnet50.py               # ResNet50 + classifier head
│   │   ├── efficientnet_b0.py        # EfficientNet-B0 + classifier head
│   │   ├── vit_b_16.py               # ViT-B/16 + classifier head
│   │   └── model_factory.py          # Factory · fallback offline · load_checkpoint_class_to_idx
│   ├── training/
│   │   └── train.py                  # Train loop · val · EMA · TTA · grad clip · early stop
│   ├── orchestration/
│   │   └── pipeline.py              # Discover data → train all → eval → ensemble → summary
│   ├── evaluation/
│   │   └── evaluate_report.py       # Metrics · CSV · comparison chart · confusion matrix
│   ├── inference/
│   │   └── inference.py             # Predict · feature hook · color/size/texture · rule-based gating
│   └── learning/
│       └── self_learning.py         # Feedback log · hard examples list
│
├── Review/
│   ├── review_terminal.py            # Round loop: train → review → đề xuất config
│   └── optimal_configs.py           # OPTIMAL_CONFIGS · TUNING_CANDIDATES  ← NGUỒN SỰ THẬT HP
│
├── models/                           # OUTPUT artifacts (không commit .pt lớn)
│   ├── *_epillid_best.pt
│   ├── *.metrics.json · *.history.json
│   ├── *_training_curves.png
│   ├── evaluation_summary.csv
│   ├── evaluation_comparison.png
│   └── reports/latest/
│       ├── evaluation_summary.json
│       └── confusion_matrix_*.png
│
├── data_aligned/                     # Dataset chuẩn — ƯU TIÊN dùng (không commit)
├── data/                             # Dataset gốc/thử nghiệm (không commit)
├── demo_images/                      # Ảnh mẫu cho demo/infer
└── tests/
    ├── test_features.py
    ├── test_inference_utils.py
    └── test_metadata.py
```

---

## CONTRACTS — ĐỌC KỸ TRƯỚC KHI SỬA CODE

> Đây là phần quan trọng nhất cho AI agent. Vi phạm contract = phá vỡ pipeline.

### Contract 1 — Dataset structure
```
data_aligned/
├── train/
│   ├── class_A/  *.jpg
│   └── class_B/  *.jpg
├── val/
│   └── ...
└── test/
    └── ...

# Bắt buộc: class folders PHẢI GIỐNG NHAU ở cả 3 split
# Nếu không: class mapping sai → accuracy ảo
```

### Contract 2 — Checkpoint naming
```
<model_name>_epillid_best.pt          # checkpoint weights
<model_name>_epillid_best.metrics.json
<model_name>_epillid_history.json

# model_name phải là: resnet50 | efficientnet_b0 | vit_b_16
# KHÔNG đặt tên khác — pipeline.py và evaluate_report.py đọc theo pattern này
```

### Contract 3 — Dataset output tuple
```python
# PillImageDataset trả về tuple 3 phần tử — không thay đổi thứ tự
(image_tensor, class_idx, image_path)
```

### Contract 4 — Class mapping safety
```python
# LUÔN dùng helper này khi load checkpoint để infer
# KHÔNG tự giả định index space của checkpoint và dataset giống nhau
from src.models.model_factory import load_checkpoint_class_to_idx

model, class_to_idx = load_checkpoint_class_to_idx(
    model_name="resnet50",
    checkpoint_path="models/resnet50_epillid_best.pt"
)
```

### Contract 5 — Report outputs
```
models/evaluation_summary.csv         # so sánh tổng hợp 3 model + ensemble
models/reports/latest/                # confusion matrix + detailed JSON
```

---

## HYPERPARAMETERS MẶC ĐỊNH

> Đã được tune qua `--mode optimize`. **Không thay đổi mà không có lý do và không ghi vào DECISIONS LOG.**

| Model | lr | weight_decay | label_smoothing | mixup_alpha | epochs | patience |
|---|---:|---:|---:|---:|---:|---:|
| ResNet50 | 6e-5 | 1.2e-3 | 0.16 | 0.35 | 28 | 6 |
| EfficientNet-B0 | 7e-5 | 1e-3 | 0.15 | 0.33 | 28 | 6 |
| ViT-B/16 | 5e-5 | 1.4e-3 | 0.20 | 0.42 | 32 | 7 |

**Nguồn sự thật:** `Review/optimal_configs.py` → `OPTIMAL_CONFIGS`

---

## TRAINING ALGORITHM — KỸ THUẬT ĐÃ IMPLEMENT

```
Transfer learning     : pretrained ImageNet weights (fallback random init nếu offline)
2-stage fine-tuning   : Stage 1 freeze backbone → Stage 2 unfreeze full
Label smoothing       : giảm over-confident, CrossEntropyLoss
Mixup                 : x' = λxᵢ + (1-λ)xⱼ,  λ ~ Beta(α, α)
Gradient clipping     : ngăn gradient explosion
EMA                   : Exponential Moving Average — làm mượt params, val ổn định hơn
TTA validation        : test-time augmentation (flip/rotate), average logits
Early stopping        : dừng khi val_acc không cải thiện hoặc train-val gap quá lớn
Optimizer             : AdamW
Scheduler             : ReduceLROnPlateau (giảm LR khi val_loss plateau)
```

---

## PATTERNS BẮT BUỘC

### Tạo model — luôn qua factory
```python
# ĐÚNG
from src.models.model_factory import create_model
model = create_model("resnet50", num_classes=8,
                     checkpoint_path="models/resnet50_epillid_best.pt")

# SAI — bypass factory, mất class mapping và fallback logic
import torchvision
model = torchvision.models.resnet50(pretrained=True)
```

### Dataset — luôn dùng PillImageDataset
```python
# ĐÚNG
from src.data.features import PillImageDataset, get_transforms
dataset = PillImageDataset(root="data_aligned/train",
                           transform=get_transforms("train"))
# output: (image_tensor, class_idx, image_path)

# SAI — mất augmentation và class_to_idx chuẩn
from torchvision.datasets import ImageFolder
dataset = ImageFolder("data_aligned/train")
```

### Config — luôn từ optimal_configs.py
```python
# ĐÚNG
from Review.optimal_configs import OPTIMAL_CONFIGS
cfg = OPTIMAL_CONFIGS["resnet50"]

# SAI — hardcode rải rác, không track được
lr = 0.0001
```

### Inference — luôn qua inference.py
```python
from src.inference.inference import predict_image
result = predict_image(
    image_path="test.jpg",
    model_name="resnet50",
    checkpoint_path="models/resnet50_epillid_best.pt"
)
# result: {"class": "...", "confidence": 0.92, "top5": [...]}
```

### So sánh 2 ảnh thuốc — kết hợp deep + heuristic
```python
# inference.py kết hợp:
# 1. Deep feature similarity (forward hook trước classifier)
# 2. Hand-crafted: color score · size score · texture score
# 3. Rule-based gating: class match + feature threshold + semantic penalty
# KHÔNG tự viết cosine similarity đơn lẻ — thiếu gating logic
```

---

## EVALUATION TARGETS

```
Mỗi model đơn lẻ  : accuracy > 85%,  macro-F1 > 0.83
Ensemble           : accuracy > 90%,  macro-F1 > 0.88
```

Nếu dưới target → xem bảng xử lý lỗi bên dưới.

---

## BOUNDARIES

### AI KHÔNG ĐƯỢC
```
❌ Khởi tạo model trực tiếp, bypass model_factory
❌ Tự viết DataLoader mà không dùng PillImageDataset
❌ Hardcode hyperparameter trong script — phải qua OPTIMAL_CONFIGS
❌ Save checkpoint với tên không theo *_epillid_best.pt
❌ Giả định class index của checkpoint và dataset giống nhau (luôn dùng load_checkpoint_class_to_idx)
❌ Thêm model mới mà không tạo file riêng trong src/models/ theo cùng interface
❌ Sửa augmentation logic ngoài features.py
❌ Commit data/ hoặc data_aligned/ lên git
❌ Commit file .pt > 100MB lên git
❌ Thay đổi hyperparameter mà không ghi vào DECISIONS LOG
```

### AI LUÔN PHẢI
```
✅ Chạy pytest -q trước khi commit thay đổi logic
✅ Kiểm tra data_aligned/ có đủ train/val/test và class folders nhất quán
✅ Dùng --device cpu khi không chắc có GPU
✅ Sau khi train: verify models/evaluation_summary.csv có đủ 3 model
✅ Khi thêm class mới: cập nhật num_classes ở CẢ 3 file model + model_factory
✅ Khi sửa transform: smoke test trước (--epochs 2 --batch-size 4)
✅ Đọc file tương tự trong src/ trước khi tạo file mới
✅ Ghi lý do vào DECISIONS LOG khi thay đổi HP hoặc kiến trúc
```

---

## XỬ LÝ LỖI THƯỜNG GẶP

| Triệu chứng | Nguyên nhân | Cách fix |
|------------|------------|---------|
| `Data split mismatch` / class không nhất quán | train/val/test có class folders khác nhau | Chạy `build_epillid_data.py` để re-align |
| `CUDA unavailable` | Không có GPU | Thêm `--device cpu` |
| Accuracy < 80% | Thiếu data hoặc LR không phù hợp | Tăng data · giảm LR · chạy `--mode optimize` |
| Pretrained download fail | Offline / không có internet | Factory tự fallback random init (kiểm tra log) |
| Thiếu report sau train | Pipeline bị interrupt | Chạy lại `python run_all.py` |
| `notebook warning google.colab` | Chạy notebook code ngoài Colab | Bỏ qua, không phải lỗi src/ |
| Git reject file .pt | File > 100MB | Thêm `models/*.pt` vào `.gitignore` |
| Val loss không giảm | LR quá cao | Giảm LR 50% · tăng patience |
| Overfitting (train >> val) | Regularization yếu | Tăng `weight_decay` · tăng `mixup_alpha` |
| Inference class sai | Class mapping không đồng bộ | Dùng `load_checkpoint_class_to_idx` |

---

## ARTIFACTS — FILE PHẢI CÓ KHI NỘP

```
models/
├── resnet50_epillid_best.pt
├── efficientnet_b0_epillid_best.pt
├── vit_b_16_epillid_best.pt
├── *.metrics.json  (3 file)
├── *.history.json  (3 file)
├── *_training_curves.png  (3 file)
├── evaluation_summary.csv
├── evaluation_comparison.png
└── reports/latest/
    ├── evaluation_summary.json
    └── confusion_matrix_*.png  (3 file)
```

Kiểm tra nhanh trước khi nộp:
```bash
python -m pytest tests/ -q                     # tests pass
python run_all.py --compare-only               # report đầy đủ
ls models/*.pt | wc -l                         # phải ra 3
ls models/reports/latest/confusion_matrix_*.png | wc -l  # phải ra 3
```

---

## KHI GẶP AMBIGUITY

```
Cần clarify trước khi implement:
1. [câu hỏi về requirement]
2. [câu hỏi về edge case]

Assumption mặc định nếu không có phản hồi:
- Dùng data_aligned/ (ưu tiên hơn data/)
- Chạy cả 3 model
- Device: tự detect GPU, fallback CPU
- Checkpoint naming: theo contract *_epillid_best.pt
```

---

## DECISIONS LOG

> Ghi lại mọi thay đổi HP hoặc kiến trúc để track regression.

| Ngày | Thay đổi | Lý do | Kết quả |
|------|---------|-------|---------|
| Mar 2026 | HP mặc định (bảng trên) | Tuned qua optimize mode | Baseline |
| | | | |

---

*AGENTS.md v2.2 — THUOC deep dive edition, March 2026*  
*Symlink cho Claude Code: `ln -s AGENTS.md CLAUDE.md`*  
*Cập nhật file này khi: thêm model/class mới · thay đổi pipeline · đổi stack*
