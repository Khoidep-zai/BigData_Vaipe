# AGENTS.md — THUOC Project
> Doc file nay TRUOC KHI viet bat ky dong code nao.
> Compatible: Claude Code · Cursor · Windsurf · GitHub Copilot Pro · Google Antigravity · Aider

---

## ROLE

Ban la ML engineer dang phat trien THUOC — he thong phan loai vien thuoc tu anh bang deep learning.

```text
Task      : Image classification (num_classes dong theo dataset/checkpoint)
Models    : ResNet50 · EfficientNet-B0 · ViT-B/16 (ensemble)
Framework : PyTorch + torchvision
Python    : 3.10+
Hardware  : CUDA GPU (auto-fallback CPU)
```

Ghi chu quan trong:
- Khong hardcode 8 classes.
- Num_classes hien tai trong data_aligned dang la 104 class folders moi split.
- Luong prescription_match su dung target_class_id 0..107 (107 = out_of_prescription).

---

## COMMANDS — CHAY TRUOC KHI LAM BAT CU DIEU GI

```bash
# Cai thu vien
pip install -r requirements.txt

# Full pipeline: train -> evaluate -> ensemble -> report  [LENH CHINH]
python run_all.py

# Chi evaluate (da co checkpoint, khong train lai)
python run_all.py --compare-only

# Dung CPU
python run_all.py --device cpu

# Train 1 model don le
python train_cli.py --mode single --model resnet50 --epochs 28
python train_cli.py --mode single --model efficientnet_b0 --epochs 28
python train_cli.py --mode single --model vit_b_16 --epochs 32

# Hyperparameter tuning nhieu vong
python train_cli.py --mode optimize --rounds 3 --epochs 12

# Smoke test / debug
python train_cli.py --mode single --model resnet50 --epochs 2 --batch-size 4

# Prescription matching
python train_cli.py --mode prescription_match --prescription-image data/images/public_train/prescription/image/VAIPE_P_TRAIN_0.png --pill-images data/images/public_train/pill/image/VAIPE_P_0_0.jpg data/images/public_train/pill/image/VAIPE_P_0_1.jpg --pretty

# Unit tests
python -m pytest tests/ -q
```

Build data tu ePillID (chi khi can import benchmark ePillID):

```bash
python src/data/build_epillid_data.py --epillid-root <duong_dan_epillid> --img-root <duong_dan_anh> --output-root data_aligned
```

---

## KIEN TRUC HE THONG

### Pipeline end-to-end

```text
[DATA]          data_aligned/ (uu tien) · cau truc split/class/*.jpg
                Bat buoc: train/ · val/ · test/ co cung class folders
        |
        v
[DATA MODULE]   PillImageDataset -> (image_tensor, class_idx, image_path)
                Train: augmentation ON  |  Val/Test: deterministic
        |
        v
[MODEL FACTORY] create_model(name) / load_checkpoint(...)
                Pretrained ImageNet -> fallback random init neu offline
        |
        v
[TRAINING]      train.py: mixup · label smoothing · EMA · TTA · grad clip
                early stop · AdamW · ReduceLROnPlateau
        |
        v
[EVALUATION]    evaluate_report.py: accuracy · macro-F1 · confusion matrix
        |
        v
[ENSEMBLE]      Weighted soft-voting theo best_val_acc
        |
        v
[OUTPUT]        models/AI + models/results + models/reports/latest
```

---

## CAU TRUC THU MUC (THUC TE)

```text
THUOC/
├── run_all.py
├── train_cli.py
├── requirements.txt
├── AGENTS.md
│
├── src/
│   ├── data/
│   │   ├── features.py
│   │   ├── metadata.py
│   │   ├── data_setup.py
│   │   ├── build_epillid_data.py
│   │   └── prescription_csv_builder.py
│   ├── models/
│   │   ├── resnet50.py
│   │   ├── efficientnet_b0.py
│   │   ├── vit_b_16.py
│   │   └── model_factory.py
│   ├── training/train.py
│   ├── orchestration/pipeline.py
│   ├── evaluation/evaluate_report.py
│   ├── inference/
│   │   ├── inference.py
│   │   └── prescription_matching.py
│   └── utils/
│       ├── model_paths.py
│       └── runtime_artifacts.py
│
├── Review/
│   ├── review_terminal.py
│   └── optimal_configs.py
│
├── models/
│   ├── AI/
│   │   ├── resnet50/
│   │   ├── efficientnet/
│   │   └── vit_b_16/
│   ├── results/
│   │   ├── evaluation/
│   │   └── training/
│   └── reports/latest/
│
├── data_aligned/
├── data/
└── tests/
```

---

## CONTRACTS — VI PHAM = PHA VO PIPELINE

### Contract 1 — Dataset structure
```text
data_aligned/train|val|test/class_name/*.jpg
Class folders phai GIONG NHAU o ca 3 split
Fix: dung data_setup.discover_or_prepare_data_dir hoac tao lai data_aligned
```

### Contract 2 — Checkpoint naming + canonical path
```text
models/AI/resnet50/resnet50_epillid_best.pt
models/AI/efficientnet/efficientnet_b0_epillid_best.pt
models/AI/vit_b_16/vit_b_16_epillid_best.pt
```

### Contract 3 — Dataset tuple output
```python
(image_tensor, class_idx, image_path)
```

### Contract 4 — Class mapping safety (quan trong)
```python
from src.models.model_factory import load_checkpoint, load_checkpoint_class_to_idx

ckpt = "models/AI/resnet50/resnet50_epillid_best.pt"
class_to_idx = load_checkpoint_class_to_idx(ckpt)
num_classes = len(class_to_idx) if class_to_idx else 104
model = load_checkpoint(
    model_name="resnet50",
    num_classes=num_classes,
    checkpoint_path=ckpt,
)
```

### Contract 5 — Model factory usage
```python
from src.models.model_factory import create_model
model, in_features = create_model(
    model_name="resnet50",
    num_classes=104,
    pretrained=True,
    fallback_to_random=True,
)
```

---

## HYPERPARAMETERS MAC DINH

Nguon su that: Review/optimal_configs.py -> OPTIMAL_CONFIGS

| Model | lr | weight_decay | label_smoothing | mixup_alpha | epochs | patience |
|---|---:|---:|---:|---:|---:|---:|
| ResNet50 | 6e-5 | 1.2e-3 | 0.16 | 0.35 | 28 | 6 |
| EfficientNet-B0 | 7e-5 | 1e-3 | 0.15 | 0.33 | 28 | 6 |
| ViT-B/16 | 5e-5 | 1.4e-3 | 0.20 | 0.42 | 32 | 7 |

---

## PATTERNS BAT BUOC

```python
# Dataset
from src.data.features import PillImageDataset, build_transforms
train_ds = PillImageDataset(
    root="data_aligned/train",
    transform=build_transforms(train=True, profile="default"),
)

# Config
from Review.optimal_configs import OPTIMAL_CONFIGS
cfg = OPTIMAL_CONFIGS["vit_b_16"]
```

```python
# Prescription matching
from src.inference.prescription_matching import match_pills_to_prescription
result = match_pills_to_prescription(
    prescription_image="data/images/public_train/prescription/image/VAIPE_P_TRAIN_0.png",
    pill_images=["data/images/public_train/pill/image/VAIPE_P_0_0.jpg"],
)
```

---

## BOUNDARIES

### AI KHONG DUOC
```text
Bypass model_factory de khoi tao/load model truc tiep
Hardcode num_classes hoac hardcode class_to_idx
Hardcode hyperparameter thay vi dung OPTIMAL_CONFIGS
Save checkpoint sai pattern *_epillid_best.pt
Viet train logic trong pipeline.py
Update weights trong evaluate_report.py
Commit data/, data_aligned/, models/*.pt len git
```

### AI LUON PHAI
```text
Chay pytest -q truoc khi commit thay doi logic
Verify train/val/test co class folders dong nhat truoc train
Sau khi train/eval, kiem tra artifact trong models/results va models/reports/latest
Doc file tuong tu trong src/ truoc khi tao file moi
Ghi ly do vao DECISIONS LOG khi doi HP hoac doi kien truc
```

---

## ARTIFACTS PHAI CO KHI NOP

```bash
python -m pytest tests/ -q
python run_all.py --compare-only
```

```text
models/
├── AI/
│   ├── resnet50/resnet50_epillid_best.pt
│   ├── efficientnet/efficientnet_b0_epillid_best.pt
│   └── vit_b_16/vit_b_16_epillid_best.pt
├── results/
│   ├── evaluation/evaluation_summary.csv
│   ├── evaluation/evaluation_comparison.png
│   ├── training/training_results_table.csv
│   └── training/training_results_table.md
└── reports/latest/
    ├── evaluation_summary.json
    └── confusion_matrix_*.png
```

---

## KHI GAP AMBIGUITY

Can clarify truoc khi implement:
1. Yeu cau output can cap nhat trong models/results hay models/reports/latest?
2. Chay compare-only tren checkpoint co san hay train lai tu dau?

Assumption mac dinh:
- Dung data_aligned (uu tien hon data/)
- Chay ca 3 model
- Device auto-detect GPU, fallback CPU
- Checkpoint naming theo *_epillid_best.pt trong models/AI/<alias>/

---

## DECISIONS LOG

| Ngay | Thay doi | Ly do | Ket qua |
|------|----------|-------|---------|
| Mar 2026 | HP mac dinh (bang tren) | Tuned qua optimize mode | Baseline |
| Mar 2026 | HP: tang regularization + giam lr | Giam train/val gap | Pending verify |
| Mar 2026 | Model heads: them Dropout(0.3) | Giam overfit | Pending verify |
| Mar 2026 | Augmentation: blur + sharpness + random erasing mo rong | Robust hon voi anh mo/sac | Pending verify |
| Mar 2026 | image_to_numeric_vector: them HSV hist + edge density + quadrant luminance | Vector thong tin phong phu hon | Pending verify |
| Mar 2026 | train.py: warmup va scheduler tinh chinh | On dinh hoi tu dau pha unfreeze | Pending verify |

---

AGENTS.md v2.5 — THUOC edition, March 2026