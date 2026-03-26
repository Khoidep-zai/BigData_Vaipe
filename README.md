# THUOC - Huong Dan Day Du

THUOC la do an phan loai vien thuoc tu anh bang deep learning.
Tai lieu nay mo ta ro: cau truc file, luong hoat dong, thuat toan train, cach chay va file can nop.

## Muc Luc Nhanh

1. Tong quan he thong
2. Cau truc thu muc va vai tro tung file
3. Luong hoat dong end-to-end
4. Giai thich thuat toan train
5. Cach su dung
6. Ket qua va artifacts
7. Colab
8. Loi thuong gap
9. Checklist nop do an

---

## Tong Quan He Thong

Pipeline cua THUOC co 6 lop:

1. Input: du lieu train/val/test + cau hinh hyperparameter.
2. Data processing: dataset, transform, dataloader.
3. Training: train 3 mo hinh ResNet50, EfficientNet-B0, ViT-B/16.
4. Evaluation: accuracy, macro-F1, confusion matrix.
5. Ensemble: bo phieu co trong so tu 3 model.
6. Output: checkpoint, metrics, reports, plots.

So do don gian:

```mermaid
flowchart LR
  A[data_aligned or data] --> B[run_all.py]
  B --> C[Train 3 models]
  C --> D[Evaluate + Ensemble]
  D --> E[CSV JSON PNG Reports]
```

---

## Cau Truc Thu Muc Va Vai Tro Tung File

```text
THUOC/
  run_all.py
  train_cli.py
  review_terminal.py
  optimal_configs.py
  requirements.txt
  THUOC_Colab_Train_Evaluate.ipynb
  src/
  data/ or data_aligned/
  models/
  tests/
```

### Entry scripts

1. run_all.py
- Entry point quan trong nhat.
- Chay train -> evaluate -> report cho 3 model.

2. train_cli.py
- CLI linh hoat, 3 mode:
- all: full pipeline
- single: train 1 model
- optimize: tuning nhieu vong

3. review_terminal.py
- Hien thi review ket qua train va goi y dieu chinh hyperparameter.

4. optimal_configs.py
- Chua OPTIMAL_CONFIGS va TUNING_CANDIDATES.

### Core modules trong src (da phan theo nhom)

1. src/training/train.py
- Training engine: train loop, validation loop, early stopping, save checkpoint.

2. src/models/resnet50.py
- Dinh nghia model ResNet50 va classifier head.

3. src/models/efficientnet_b0.py
- Dinh nghia model EfficientNet-B0 va classifier head.

4. src/models/vit_b_16.py
- Dinh nghia model ViT-B/16 va classifier head.

5. src/models/model_factory.py
- Tao model theo ten va load checkpoint/class mapping.

6. src/data/features.py
- PillImageDataset, transforms, augmentation, class_to_idx.

7. src/orchestration/pipeline.py
- Orchestration: discover data dir, train all, evaluate all, ensemble.

8. src/inference/inference.py
- Load model, predict mot anh, confidence score, class mapping safety.

9. src/evaluation/evaluate_report.py
- Tinh metrics va xuat report (csv/json/png).

10. src/data/build_epillid_data.py
- Build/align du lieu train-val-test tu nguon raw.

11. src/data/metadata.py
- Parse metadata csv cho thong tin thuoc.

12. src/learning/self_learning.py
- Ghi feedback va hard examples de cai thien ve sau.

---

## Luong Hoat Dong End-to-End

1. Phat hien du lieu
- Uu tien data_aligned, neu khong co thi dung data.
- Bat buoc co train, val, test.

2. Tao dataset va dataloader
- PillImageDataset doc anh va nhan.
- Train co augmentation, val/test khong random augmentation.

3. Khoi tao model
- Chon 1 trong 3 model.
- Load pretrained weights (neu co), sau do thay classifier cho 8 classes.

4. Train theo epoch
- Train phase: forward, loss, backward, optimizer step.
- Validation phase: tinh val_loss, val_acc.
- Luu best checkpoint khi val_acc cai thien.

5. Early stopping
- Dung neu gap train-val qua lon hoac het patience.

6. Evaluate test
- Tinh accuracy, macro-F1.
- Ve confusion matrix tung model.

7. Ensemble
- Tong hop logits/bo phieu co trong so tu 3 model.
- Tinh metrics ensemble.

8. Xuat artifacts
- models/*.pt, *.metrics.json, *.history.json
- models/evaluation_summary.csv
- models/reports/latest/*

---

## Giai Thich Thuat Toan Train

### Kien truc model

1. ResNet50
- Residual blocks voi skip-connection.
- Manh, on dinh, tham so lon.

2. EfficientNet-B0
- Nhe, nhanh, can bang tot toc do va do chinh xac.

3. ViT-B/16
- Transformer cho vision, hoc global context qua attention.

### Loss va regularization

1. CrossEntropyLoss
- Loss co ban cho classification.

2. Label smoothing
- Lam mem nhan de giam over-confident.

3. Mixup
- Tron 2 mau trong batch de tang kha nang tong quat hoa.

4. Weight decay
- L2 regularization, giam overfitting.

### Optimizer va scheduler

- Optimizer: AdamW.
- Scheduler: ReduceLROnPlateau.
- Learning rate giam khi val_loss khong cai thien.

### Strategy train 2 giai doan

1. Stage 1: freeze backbone
- Train classifier head de on dinh ban dau.

2. Stage 2: unfreeze full model
- Fine-tune toan bo mang voi LR phu hop.

### Vi sao model co the hoc nhanh

- Transfer learning tu pretrained ImageNet.
- Dataset co cau truc class ro rang theo folder.
- Regularization + early stop han che overfit.

---

## Hyperparameter Mac Dinh

| Model | lr | weight_decay | label_smoothing | mixup_alpha | epochs | patience |
|---|---:|---:|---:|---:|---:|---:|
| ResNet50 | 6e-5 | 1.2e-3 | 0.16 | 0.35 | 28 | 6 |
| EfficientNet-B0 | 7e-5 | 1e-3 | 0.15 | 0.33 | 28 | 6 |
| ViT-B/16 | 5e-5 | 1.4e-3 | 0.20 | 0.42 | 32 | 7 |

---

## Cach Su Dung

### Cai thu vien

```bash
pip install -r requirements.txt
```

### Chay full pipeline (khuyen dung)

```bash
python run_all.py
```

### Lenh hay dung

```bash
# train 1 model
python train_cli.py --mode single --model resnet50 --epochs 28

# optimize nhieu vong
python train_cli.py --mode optimize --rounds 3 --epochs 12

# chi evaluate model da train
python run_all.py --compare-only

# dung CPU
python run_all.py --device cpu
```

---

## Ket Qua Va Artifacts

Ket qua duoc luu trong thu muc models:

1. Checkpoint
- resnet50_epillid_best.pt
- efficientnet_b0_epillid_best.pt
- vit_b_16_epillid_best.pt

2. Metrics/history
- *_epillid_best.metrics.json
- *_epillid_history.json

3. Visualization
- *_training_curves.png
- evaluation_comparison.png
- confusion_matrix_*.png

4. Report tong hop
- models/evaluation_summary.csv
- models/reports/latest/evaluation_summary.json

---

## File Quan Trong De Nop Do An

1. Source code: src/ day du.
2. 3 checkpoint: models/*_epillid_best.pt.
3. models/evaluation_summary.csv.
4. models/evaluation_comparison.png.
5. 3 file training curves.
6. confusion_matrix_*.png trong models/reports/latest.
7. README.md va requirements.txt.

---

## Chay Tren Google Colab

File notebook: THUOC_Colab_Train_Evaluate.ipynb

Cac buoc:

1. Mo notebook tren Colab.
2. Sua REPO_URL va DRIVE_DATA_ROOT o cell cau hinh.
3. Chay lan luot cac cell setup -> train -> evaluate -> package.

Loi ich:
- Dung GPU Colab de train nhanh.
- Tu dong tao artifacts va report.
- Co the zip output de tai ve/luu Drive.

---

## Kiem Thu Nhanh

```bash
python -m pytest tests/ -q
```

Functional test nhanh:

```bash
python train_cli.py --mode single --model resnet50 --epochs 2 --batch-size 4
```

---

## Loi Thuong Gap Va Cach Xu Ly

1. Khong tim thay du lieu
- Kiem tra data_aligned/data co train, val, test.

2. CUDA khong kha dung
- Chay bang CPU: python run_all.py --device cpu.

3. Accuracy thap
- Tang du lieu moi class.
- Giam LR, tang epochs, chay optimize mode.

4. Thieu report
- Chay lai full pipeline: python run_all.py.

5. Push Git bi loi file lon
- Ignore checkpoint .pt neu can.

---

## Checklist Truoc Khi Nop

1. Da co 3 checkpoint trong models.
2. Da co evaluation_summary.csv va evaluation_comparison.png.
3. Da co confusion matrix trong models/reports/latest.
4. Da co training curves cho 3 model.
5. Da chay test: python -m pytest tests/ -q.
6. README + requirements di kem source.
7. Kiem tra git status sach truoc khi nop.

---

Last Updated: March 2026
