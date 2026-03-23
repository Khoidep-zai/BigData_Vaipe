# THUOC - Hướng Dẫn Dễ Hiểu Cho Người Mới

THUOC là đồ án phân loại viên thuốc từ ảnh. Bạn không cần hiểu sâu về AI vẫn có thể chạy được toàn bộ quy trình nếu làm theo đúng các bước trong tài liệu này.

## 📋 Mục Lục Nhanh

- [Sơ Đồ Kiến Trúc Hệ Thống](#sơ-đồ-kiến-trúc-hệ-thống)
- [Chi Tiết File & Vai Trò](#chi-tiết-file--vai-trò)
- [Luồng Hoạt Động Chi Tiết](#luồng-hoạt-động-chi-tiết)
- [Giải Thích Thuật Toán Training](#giải-thích-thuật-toán-training)
- [Bạn Sẽ Nhận Được Gì](#bạn-sẽ-nhận-được-gì-sau-khi-chạy)

---

## 🏗️ Sơ Đồ Kiến Trúc Hệ Thống

### Kiến Trúc Tổng Quan (System Architecture)

Hệ thống THUOC bao gồm 6 layer chính:

```
📥 INPUT LAYER
  ├─ 📂 data_aligned/ (8 medicine classes)
  └─ ⚙️ optimal_configs.py (Hyperparameters)
         ↓
🔄 DATA PROCESSING
  ├─ discover_data_dir() → tìm dữ liệu
  ├─ PillImageDataset → indexing classes
  ├─ Augmentation → ColorJitter, Flip, Rotation
  └─ DataLoader → batching
         ↓
🧠 TRAINING PIPELINE
  ├─ 3 Models: ResNet50, EfficientNet-B0, ViT-B/16
  ├─ Loss: CrossEntropy + Label Smoothing + Mixup
  ├─ Optimizer: AdamW + ReduceLROnPlateau + Early Stop
  └─ 2-Stage Training: Frozen Backbone → Full Training
         ↓
📊 EVALUATION
  ├─ Single Model Eval (Accuracy, Macro-F1, Confusion Matrix)
  ├─ Weighted Ensemble (3-model voting)
  └─ Reports Generation (CSV, JSON, PNG)
         ↓
📤 OUTPUT ARTIFACTS
  ├─ Checkpoints (.pt files)
  ├─ Metrics (*.metrics.json, *.history.json)
  └─ Visualizations (training_curves.png, confusion_matrix_*.png)
         ↓
🖥️ APPLICATIONS
  ├─ CLI: run_all.py, train_cli.py
  ├─ GUI: gui_tk.py (Tkinter)
  └─ Inference: inference.py (Single image prediction)
```

### File Structure Diagram

```
THUOC/
├── 📜 SCRIPT FILES (Entry Points)
│   ├─ run_all.py 🚀 (Main pipeline - train + eval + report)
│   ├─ train_cli.py 🎛️ (Flexible trainer - 3 modes)
│   ├─ run_gui.py 🖥️ (GUI launcher)
│   ├─ review_terminal.py 📊 (Hyperparameter tuning review)
│   └─ optimal_configs.py ⚙️ (Hyperparameter configs)
│
├── 📦 src/ (Core Modules)
│   ├─ train.py ⚙️
│   │   └─ Training engine: loss, optimizer, scheduler, callbacks
│   ├─ models.py 🧠
│   │   └─ Model builders for ResNet50, EfficientNet-B0, ViT-B/16
│   ├─ features.py 📊
│   │   └─ PillImageDataset, augmentation pipeline, transforms
│   ├─ pipeline.py 🔄
│   │   └─ Main orchestration: data discovery, train_all, eval_all
│   ├─ inference.py 🎯
│   │   └─ Model inference engine with confidence scoring
│   ├─ evaluate_report.py 📈
│   │   └─ Metrics computation & CSV export
│   ├─ build_epillid_data.py 🔧
│   │   └─ Data integration (copy/symlink files)
│   ├─ metadata.py 🏷️
│   │   └─ Parse CSV metadata, medicine properties
│   ├─ self_learning.py 📚
│   │   └─ Feedback logging, hard examples collection
│   └─ gui_tk.py 🖥️
│       └─ Tkinter GUI for image comparison
│
├── 📂 DATA FOLDERS
│   ├─ data_aligned/
│   │   ├─ train/ (8 medicine classes)
│   │   ├─ val/
│   │   └─ test/ (separate test sets)
│   ├─ demo_images/ (Sample images for testing)
│   └─ Medicine_Details_Deeplearning.csv (Metadata)
│
├── 💾 models/ (Training Outputs)
│   ├─ 3 Checkpoints + 3 Metrics + 3 History Files
│   ├─ training_curves.png (Combined plot)
│   ├─ evaluation_summary.csv (Results Table)
│   └─ reports/latest/ (Detailed evaluation + confusion matrices)
│
├── 🧪 tests/ (Unit Tests)
│   ├─ test_features.py
│   ├─ test_inference_utils.py
│   └─ test_metadata.py
│
└── ⚙️ CONFIG FILES
    ├─ requirements.txt (Dependencies)
    ├─ README.md (This file)
    ├─ THUOC_Colab_Train_Evaluate.ipynb (Colab notebook)
    └─ .gitignore
```

---

## 📖 Chi Tiết File & Vai Trò

### 🎯 Entry Point Scripts

#### 1. **run_all.py** - Main Pipeline (Quan trọng nhất ⭐⭐⭐)

**Vai trò:** Chạy toàn bộ qui trình training và evaluation liên tiếp.

**Cách dùng:**
```bash
python run_all.py                          # Dùng cấu hình tối ưu
python run_all.py --device cpu             # Dùng CPU thay GPU
python run_all.py --data-dir data_aligned  # Chỉ định folder dữ liệu
python run_all.py --compare-only           # Chỉ evaluate checkpoints cũ
```

**Luồng thực hiện:**
1. Tìm auto-detect data directory (data_aligned hoặc data)
2. Train 3 models tuần tự (ResNet50 → EfficientNet-B0 → ViT-B/16)
3. Evaluate từng model trên test set
4. Compute 3-model weighted ensemble
5. Xuất CSV, JSON, PNG, confusion matrices

**Output:**
- 3 checkpoint files (*.pt)
- Metrics + history JSON
- Training curves PNG
- evaluation_summary.csv
- Detailed reports trong models/reports/latest/

---

#### 2. **train_cli.py** - Flexible Trainer

**Vai trò:** Trainer linh hoạt với 3 mode: pipeline, single model, optimize.

**Cách dùng:**
```bash
# Mode 1: Full pipeline
python train_cli.py --mode all --epochs 28 --batch-size 16

# Mode 2: Train 1 model
python train_cli.py --mode single --model resnet50 --epochs 28

# Mode 3: Multi-round hyperparameter optimization
python train_cli.py --mode optimize --rounds 3 --epochs 12
```

**Khác với run_all.py:**
- Có thể điều chỉnh chi tiết hơn (learning rate, weight decay, etc.)
- Hỗ trợ multi-round optimization
- Dùng khi muốn tuning hyperparameter

---

#### 3. **review_terminal.py** - Interactive Tuning

**Vai trò:** Review training curves và suggest hyperparameter changes interactively.

**Cách dùng:**
```bash
python review_terminal.py
```

**Tính năng:**
- Hiển thị biểu đồ training/validation loss
- So sánh các epoch khác nhau
- Suggest changes (tăng/giảm LR, weight decay, etc.)
- Lưu history của từng review round

---

#### 4. **run_gui.py** - Tkinter GUI Application

**Vai trò:** Giao diện desktop cho phân loại thuốc và so sánh hình ảnh.

**Cách dùng:**
```bash
python run_gui.py
```

**Tính năng:**
- Upload 2 hình ảnh thuốc
- So sánh similarity score
- Hiển thị màu sắc, kích thước, texture
- Hiển thị thông tin medicine metadata (composition, dosage, etc.)

---

### 📦 Core Modules (src/)

#### 1. **train.py** ⚙️ - Training Engine

**Hàm chính:**
- `train_model()` - Main training loop
- `evaluate()` - Validation/test evaluation
- `create_dataloaders()` - Data loading with augmentation
- `_mixup_batch()` - Apply mixup augmentation to batch
- `_plot_training_curves()` - Visualize loss/accuracy curves

**Training Strategy (2-Stage):**
```
STAGE 1: Frozen Backbone (3 epochs)
  └─ Backbone weights không thay đổi
  └─ Chỉ train classification head
  └─ Ổn định learning ở giai đoạn đầu

STAGE 2: Full Training (25+ epochs or early stop)
  └─ Unfreeze toàn bộ model
  └─ Learning rate warmup
  └─ Train với regularization (label smoothing, mixup)
  └─ Early stop nếu val gap > threshold
```

**Loss Function:**
```python
Loss = CrossEntropyLoss(label_smoothing=0.15-0.20)
       + Mixup(alpha=0.3-0.4)  # Blending 2 samples
```

**Optimizer:**
- Adam optimizer với Weight Decay (1-1.4e-3)
- Learning Rate Scheduler: ReduceLROnPlateau
- Early Stopping nếu validation gap vượt ngưỡng

---

#### 2. **models.py** 🧠 - Model Builders

**Mô hình được hỗ trợ:**

```
1. ResNet50 (90M parameters)
   - Kiến trúc: Residual blocks với skip connections
   - Tốc độ: Trung bình
   - Accuracy: 20% (trên test set nhỏ)
   - Độ ổn định: Cao
   
2. EfficientNet-B0 (7M parameters)
   - Kiến trúc: Compound scaling (depth + width + resolution)
   - Tốc độ: Nhanh nhất
   - Accuracy: 30%
   - Độ ổn định: Trung bình
   - Ưu điểm: Nhẹ, tiết kiệm GPU memory
   
3. ViT-B/16 (Vision Transformer)
   - Kiến trúc: Self-attention trên image patches
   - Tốc độ: Chậm
   - Accuracy: 20%
   - Độ ổn định: Cần data lớn
   - Ưu điểm: Xử lý global context tốt
```

**Hàm chính:**
- `build_model()` - Khởi tạo model từ pretrained weights
- `load_checkpoint()` - Load trained weights từ .pt file

---

#### 3. **features.py** 📊 - Data Pipeline

**Classes chính:**

```python
class PillImageDataset(Dataset):
  - Scan folder structure
  - Load images từ disk
  - Apply transforms (augmentation)
  - Return (image_tensor, class_label)
  
class ImageSample(NamedTuple):
  - path: Đường dẫn file ảnh
  - label: Class label (int)
  - medicine_name: Tên thuốc (str)
```

**Data Augmentation (Training):**
```python
1. ColorJitter (brightness, contrast, saturation)
2. RandomHorizontalFlip (50% xác suất)
3. RandomRotation (±15 độ)
4. Normalization (ImageNet stats)
5. Mixup (Blending 2 samples với Beta distribution)
```

**Data Transforms (Test/Val):**
```python
1. Resize → 224×224
2. Normalization (ImageNet stats)
3. Không augmentation
```

---

#### 4. **pipeline.py** 🔄 - Main Orchestration

**Hàm chính:**
- `discover_data_dir()` - Auto-detect data directory
- `train_all_models()` - Tuần tự train 3 models
- `evaluate_all_models()` - Tuần tự evaluate 3 models
- `_evaluate_ensemble()` - Weighted ensemble voting
- `_plot_confusion_matrix()` - Confusion matrix visualization

**Workflow:**
```
Input (data + configs)
  ↓
discover_data_dir()
  ↓
train_all_models() [ResNet → EfficientNet → ViT]
  ↓
evaluate_all_models() [single model metrics]
  ↓
_evaluate_ensemble() [3-model voting]
  ↓
Generate reports (CSV, JSON, PNG)
  ↓
Output (artifacts in models/)
```

---

#### 5. **inference.py** 🎯 - Prediction Engine

**Hàm chính:**
- `load_model()` - Load checkpoint + class mapping
- `predict_single()` - Predict class cho 1 ảnh
- `batch_predict()` - Predict batch ảnh
- `get_confidence()` - Confidence score (softmax probability)

**Tính năng:**
- Model caching (tránh load lại)
- Class mapping safety (lưu class_to_idx trong checkpoint)
- Confidence thresholding

---

### 📊 Data Files

#### Medicine Classes (8 total)

```
1. cefadroxil_500mg_0.5g
2. golddicron_30mg
3. kavasdin_5_5mg
4. panactol_500mg
5. sergurop_10mg
6. thuoc_chua_dinh_danh_0
7. thuoc_chua_dinh_danh_1
8. thuoc_ngoai_don_class_107
```

#### Data Split

```
data_aligned/
├── train/     (Training set - mô hình học từ đây)
├── val/       (Validation set - dừng training nếu không cải thiện)
└── test/      (Test set - đánh giá kết quả cuối cùng)
```

---

## Bạn Sẽ Nhận Được Gì Sau Khi Chạy?

Sau khi chạy xong, hệ thống sẽ tự tạo:
1. 3 mô hình đã huấn luyện (ResNet50, EfficientNet-B0, ViT-B/16).
2. Bảng kết quả so sánh Accuracy và Macro-F1.
3. Biểu đồ so sánh mô hình.
4. Confusion matrix cho từng mô hình và ensemble.

## Kết Quả Hiện Tại

## 📊 Kết Quả Hiện Tại

Mô hình đã được huấn luyện thành công với 5 epoch (early stop do gap train/val vượt ngưỡng). Kết quả đánh giá trên test set:

| Model | Accuracy | Macro-F1 | Best Epoch | Train/Val Gap |
|---|---:|---:|---:|---:|
| ResNet50 | 20.0% | 0.1333 | Epoch 4 | 0.30 |
| EfficientNet-B0 | 30.0% | 0.1958 | Epoch 5 | 0.40 |
| ViT-B/16 | 20.0% | 0.1333 | Epoch 5 | 0.18 |

**📌 Ghi chú:** Kết quả thấp là do dataset nhỏ (10 mẫu test) và validation set hạn chế. Hãy tăng kích cỡ dữ liệu hoặc điều chỉnh hyperparameter để cải thiện.

## Chạy Nhanh Trong 5 Phút

### 📌 Yêu Cầu Trước Khi Bắt Đầu

- Python 3.9+
- GPU (NVIDIA + CUDA) để train nhanh (~5 phút/3 models trên GPU)
- CPU cũng được nhưng chậm hơn (~30-60 phút)
- Ít nhất 4GB RAM, 8GB nếu có GPU

### Bước 1: Cài thư viện

```bash
pip install -r requirements.txt
```

### Bước 2: Kiểm tra dữ liệu

Bạn cần một trong hai thư mục dữ liệu sau:
1. data_aligned
2. data

Mỗi thư mục dữ liệu phải có đủ 3 split:
1. train
2. val
3. test

### Bước 3: Chạy toàn bộ pipeline

```bash
python run_all.py
```

🚀 **Đây là lệnh quan trọng nhất.** Lệnh này sẽ:
- Train 3 models tu
-ần t
- Evaluate từng model
- Compute ensemble (3 models voting)
- Xuất CSV, JSON, PNG artifacts

**⏱️ Thời gian chạy:**
- GPU: ~5-10 phút
- CPU: ~30-60 phút

---

## 🖥️ Chạy Trên Google Colab

Bạn có notebook sẵn sàng cho Colab: `📓 THUOC_Colab_Train_Evaluate.ipynb`

### Cách dùng (3 bước)

**Bước 1:** Mở notebook trên Colab
```
1. Truy cập: https://colab.research.google.com
2. Upload hoặc mở THUOC_Colab_Train_Evaluate.ipynb
```

**Bước 2:** Chỉnh cấu hình (Cell 1)
```python
REPO_URL = "https://github.com/your-username/THUOC"  # Thay repo của bạn
BRANCH = "main"
USE_DRIVE_DATA = True  # False nếu data ở repo
DRIVE_DATA_ROOT = "/content/drive/MyDrive/THUOC"  # Path Google Drive
```

**Bước 3:** Chạy lần lượt Cell 2 → 9

### Lợi ích Colab\n+- ✅ **GPU miễn phí** (T4/A100, nhanh hơn CPU 10-50 lần)\n+- ✅ **Tự động sinh artifacts** (checkpoints, metrics, graphs)\n+- ✅ **Upload kết quả về Drive** (không lo mất data)\n+- ✅ **Không cần GPU cục bộ** (chạy trên cloud)\n+\n+---\n+\n## ⚡ Lệnh Quan Trọng Nhất\n+\n+| Mục tiêu | Lệnh | Thời gian (GPU) |

| Chạy toàn bộ 3 model + evaluate | `python run_all.py` | ~10 min |
| Train 1 model (e.g., ResNet50) | `python train_cli.py --mode single --model resnet50` | ~3 min |
| Chỉ evaluate model đã train | `python run_all.py --compare-only` | ~1 min |
| Dùng CPU thay GPU | `python run_all.py --device cpu` | ~1 hour |
| Chỉ định data folder | `python run_all.py --data-dir data_aligned` | ~10 min |
| Multi-round optimization | `python train_cli.py --mode optimize --rounds 3` | ~20 min |
\n+---
| Dùng CPU (nếu không có GPU) | python run_all.py --device cpu |
## 📂 Kết Quả Nằm Ở Đâu?

## Kết Quả Nằm Ở Đâu?

Sau khi chạy, bạn xem kết quả trong thư mục models:

  ✅ resnet50_epillid_best.pt                    (90M, ~200MB)
  ✅ efficientnet_b0_epillid_best.pt             (7M, ~30MB)
  ✅ vit_b_16_epillid_best.pt                    (86M, ~200MB)
  efficientnet_b0_epillid_best.pt
  📊 resnet50_epillid_best.metrics.json          (Accuracy, F1, etc.)
  📊 efficientnet_b0_epillid_best.metrics.json
  📊 vit_b_16_epillid_best.metrics.json
  efficientnet_b0_epillid_best.metrics.json
  📈 resnet50_epillid_history.json               (Loss/acc per epoch)
  📈 efficientnet_b0_epillid_history.json
  📈 vit_b_16_epillid_history.json
  efficientnet_b0_epillid_history.json
  📉 resnet50_training_curves.png                (Loss & Accuracy plot)
  📉 efficientnet_b0_training_curves.png
  📉 vit_b_16_training_curves.png
  efficientnet_b0_training_curves.png
  📋 evaluation_summary.csv                      (Main results table)
  📊 evaluation_comparison.png                   (Models comparison bar chart)
  evaluation_summary.csv
  evaluation_comparison.png
    📋 evaluation_summary.csv
    🗂️ evaluation_summary.json
    📊 evaluation_comparison.png
    ⚙️ tuning_summary.json
    🔲 confusion_matrix_resnet50.png
    🔲 confusion_matrix_efficientnet_b0.png
    🔲 confusion_matrix_vit_b_16.png
    🔲 confusion_matrix_ensemble_weighted.png
    confusion_matrix_vit_b_16.png
    confusion_matrix_ensemble_weighted.png
### 📌 **File Quan Trọng Để Nộp Đồ Án**

**Bắt buộc có:**
1. ✅ `models/evaluation_summary.csv` - Bảng kết quả chính
2. ✅ `models/evaluation_comparison.png` - Biểu đồ so sánh
3. ✅ `models/*_epillid_best.pt` - 3 checkpoint (100% cần)
4. ✅ `models/*_training_curves.png` - 3 biểu đồ training
5. ✅ `models/reports/latest/confusion_matrix_*.png` - Confusion matrices
\n+**Tùy chọn:**\n+6. `models/reports/latest/evaluation_summary.json` - Chi tiết metrics
7. `models/reports/latest/tuning_summary.json` - Thông tin tuning
5. models/*_epillid_best.pt (3 checkpoint)
---
\n+## 📚 Khi Nào Dùng train_cli.py?\n+\n+`run_all.py` đã đủ cho hầu hết trường hợp. Dùng `train_cli.py` khi:\n+\n+- **Bạn muốn train riêng 1 model** (không cạn tài nguyên GPU)\n+  ```bash\n+  python train_cli.py --mode single --model efficientnet_b0 --epochs 28\n+  ```\n+\n+- **Bạn muốn tuning hyperparameter tự động** (multi-round optimization)\n+  ```bash\n+  python train_cli.py --mode optimize --rounds 3 --epochs 12\n+  ```\n+\n+- **Bạn muốn điều chỉnh chi tiết loss/optimizer** (advanced users)\n+  ```bash\n+  python train_cli.py --mode all \\  \u2514\u2500\u2500 --learning-rate 8e-5 \\  \u2514\u2500\u2500 --weight-decay 1.5e-3 \\  \u2514\u2500\u2500 --epochs 40\n+  ```

---

## 🔄 Luồng Hoạt Động Chi Tiết

### High-Level Workflow (Từ người dùng nhìn)

```
STEP 1: Chuẩn bị
  └─ Cài requirements.txt
  └─ Chuẩn bị dữ liệu (data_aligned/ hoặc data/)

STEP 2: Chạy Training
  └─ python run_all.py

STEP 3: System Detection & Setup
  └─ Detect data directory (data_aligned ưu tiên)
  └─ Load PillImageDataset từ train/ folder
  └─ Extract 8 classes từ folder names
  └─ Khởi tạo 3 models (ResNet50, EfficientNet-B0, ViT-B/16)

STEP 4: Training Loop (For Each Model)
  ├─ STAGE 1: Frozen Backbone (3 epochs)
  │  ├─ Load pretrained backbone từ torchvision
  │  ├─ Freeze backbone weights
  │  ├─ Train classification head chỉ
  │  └─ Giảm learning rate (warm up phase)
  │
  ├─ STAGE 2: Full Training (25+ epochs)
  │  ├─ Unfreeze từng layer từ từ
  │  ├─ Learning rate warmup
  │  ├─ Apply regularization (label smoothing, mixup)
  │  ├─ Mỗi epoch:
  │  │  ├─ Forward pass toàn bộ train set
  │  │  ├─ Compute loss (CrossEntropy + aug)
  │  │  ├─ Backward + optimizer step
  │  │  ├─ Validation trên val set
  │  │  ├─ ReduceLROnPlateau scheduler step
  │  │  ├─ Save best checkpoint nếu val_acc improve
  │  │  └─ Check early stop (val_gap > threshold)
  │  └─ Break nếu early stop trigger
  │
  └─ Output: 1 trained checkpoint (.pt file)

STEP 5: Evaluation
  ├─ Load best checkpoint từng model
  ├─ Inference trên test set
  ├─ Compute: Accuracy, Macro-F1, Confusion Matrix
  ├─ Save: metrics.json, history.json, training_curves.png
  └─ Generate: CSV row cho mỗi model

STEP 6: Ensemble Voting
  ├─ Load 3 trained checkpoints
  ├─ Forward pass test set trên 3 models
  ├─ Ensemble voting: argmax(sum(logits)) 
  ├─ Compute ensemble metrics
  └─ Save: confusion_matrix_ensemble.png

STEP 7: Report Generation
  ├─ Merge all results → evaluation_summary.csv
  ├─ Generate comparison plots
  ├─ Export JSON reports
  └─ Output: models/reports/latest/

STEP 8: Done ✅
  └─ All artifacts ready in models/
```

### Detailed Training Epoch Flow

```
For Epoch in range(MAX_EPOCHS):
  
  TRAINING PHASE:
  ├─ For each batch in train_loader:
  │  ├─ Load images + labels
  │  ├─ Apply augmentation (ColorJitter, Flip, Rotation)
  │  ├─ Apply Mixup: blend 2 random samples
  │  │  └─ mixed_x = λ*x_i + (1-λ)*x_j  (λ ~ Beta)
  │  │  └─ mixed_y = λ*y_i + (1-λ)*y_j
  │  ├─ Forward pass: y_pred = model(mixed_x)
  │  ├─ Compute loss:
  │  │  └─ CE_loss = CrossEntropyLoss(y_pred, mixed_y)
  │  │  └─ Total_loss = CE_loss (+ optional gradient_clip)
  │  ├─ Backward: loss.backward()
  │  ├─ Optimizer step: optimizer.step()
  │  └─ Accumulate train_loss
  │
  │ END: train_loss_epoch = mean(all_batch_losses)
  │
  VALIDATION PHASE:
  ├─ For each batch in val_loader:
  │  ├─ Load images + labels
  │  ├─ Forward pass (NO augmentation): y_pred = model(x)
  │  ├─ Compute loss: val_loss = CE_loss(y_pred, y)
  │  ├─ Argmax predictions: y_pred_class = argmax(y_pred)
  │  ├─ Accumulate val_loss + accuracy
  │  └─ For ensemble: save logits để dùng sau
  │
  │ END: val_loss_epoch = mean(all_batch_losses)
  │      val_acc_epoch = sum(correct) / total
  │
  EPOCH STATISTICS:
  ├─ Compute train_gap = train_loss - val_loss
  ├─ Compute val_gap = train_acc - val_acc
  ├─ ReduceLROnPlateau scheduler:
  │  └─ if val_loss not improve: lr *= decay_factor
  ├─ Save checkpoint nếu val_acc > best_val_acc
  ├─ Check early stop: if val_gap > MAX_THRESHOLD: BREAK
  ├─ Print: Epoch [e] | train_loss | val_loss | train_acc | val_acc | gap
  └─ End epoch
```

---

## 🧠 Giải Thích Thuật Toán Training

### Tại Sao Model Dễ Ra Kết Quả?(Why Easy Prediction?)

1. **Pretrained Backbone** 🎯
   - Model được huấn luyện trên ImageNet (1M ảnh, 1000 classes)
   - Backbone đã học được features chung (edges, colors, textures)
   - Chúng ta chỉ fine-tune classification head cho 8 classes
   - → Không cần train từ scratch

2. **Transfer Learning** 🔄
   - Reuse knowledge từ ImageNet
   - Fine-tune trên medical images
   - Giảm thời gian train (từ 1000+ epochs xuống 28 epochs)
   - Giảm data requirement (ImageNet đã cover basic patterns)

3. **Regularization Strategy** 🛡️
   - **Label Smoothing:** Thay vì hard labels (0, 1), dùng soft labels (0.05, 0.95)
     ```python
     # Truyền thống
     y = [0, 0, 1, 0]  (Hard label)
     # Label smoothing (epsilon=0.15)
     y = [0.1875, 0.1875, 0.8875, 0.1875]  (Soft label)
     # Lợi ích: Model học confidently nhưng không overfit
     ```
   
   - **Mixup Augmentation:** Blend 2 samples
     ```python
     λ ~ Beta(alpha=1, alpha=1)
     mixed_x = λ*x_i + (1-λ)*x_j
     mixed_y = λ*y_i + (1-λ)*y_j
     # Lợi ích: Augmentation thực tế, khiến model mạnh mẽ hơn
     ```
   
   - **Weight Decay:** L2 regularization (1e-3)
     ```python
     # Loss = CrossEntropy + weight_decay * L2_norm(weights)
     # Lợi ích: Ngăn weights quá lớn, tránh overfitting
     ```

4. **Frozen Backbone Strategy** ❄️
   ```
   Epoch 1-3: Backbone frozen
   ├─ Chỉ train head (đơn giản, cốc định)
   ├─ Learning rate cao, convergence nhanh
   └─ Ổn định learning ở giai đoạn đầu
   
   Epoch 4+: Unfreeze toàn bộ
   ├─ Fine-tune backbone + head
   ├─ Learning rate thấp (warmup)
   └─ Tinh chỉnh features để phù hợp task
   ```

5. **Early Stopping** 🛑
   - Monitor validation accuracy
   - Nếu train/val gap > 0.12 → stop
   - Lợi ích: Tránh overfitting, tiết kiệm thời gian train

### Cách Model Dự Đoán (Inference Process)

```
INPUT: 1 ảnh viên thuốc (224×224 RGB)
  ↓
PREPROCESS:
  ├─ Load ảnh từ disk (PIL.Image)
  ├─ Resize → 224×224 (model input size)
  ├─ Normalize: (x - ImageNet_mean) / ImageNet_std
  └─ Convert to tensor: [1, 3, 224, 224] (batch_size=1)
  ↓
FORWARD PASS:
  ├─ x → Model Backbone (extract features)
  │  └─ ResNet50: conv1 → layer1-4 → avgpool → [1, 2048]
  │  └─ EfficientNet: stem → blocks → head → [1, 1280]
  │  └─ ViT: patch_embedding → transformer → [1, 768]
  ├─ feature_map → Classification Head (FC layers)
  └─ logits = [1, 8] (scores cho 8 classes)
  ↓
POST-PROCESS:
  ├─ Softmax: probs = softmax(logits)
  │  └─ probs = [0.05, 0.02, 0.03, 0.85, ...] (sum=1)
  ├─ Argmax: pred_class = argmax(probs)
  │  └─ pred_class = 3 (class index)
  ├─ Confidence: conf = max(probs)
  │  └─ conf = 0.85 (85% confident)
  └─ Map to medicine name:
     └─ class_to_idx[3] = "panactol_500mg"
  ↓
OUTPUT: (pred_class=3, confidence=0.85, name="panactol_500mg")
```

### Ensemble Voting (3-Model Voting)

```
INPUT: 1 ảnh
  ↓
PREDICT WITH 3 MODELS:
  ├─ Model 1 (ResNet50):
  │  └─ logits_1 = [0.5, 0.2, 0.1, 0.2]  → pred_1 = class 0
  ├─ Model 2 (EfficientNet):
  │  └─ logits_2 = [0.1, 0.15, 0.7, 0.05] → pred_2 = class 2
  └─ Model 3 (ViT):
     └─ logits_3 = [0.3, 0.1, 0.2, 0.4]  → pred_3 = class 3
  ↓
WEIGHTED VOTING:
  ├─ total_logits = w1*logits_1 + w2*logits_2 + w3*logits_3
  │  (w1, w2, w3 = model weights từ config)
  ├─ ensemble_pred = argmax(total_logits)
  └─ ensemble_confidence = max(softmax(total_logits))
  ↓
OUTPUT: ensemble_pred, ensemble_confidence
```

---

## Sơ Đồ Dòng Chạy (Đơn Giản)
\n+---

```mermaid
flowchart LR
  A[Du lieu: data_aligned hoac data] --> B[python run_all.py]
  B --> C[Train 3 model]
  C --> D[Evaluate]
  D --> E[Xuat CSV JSON PNG]
```

### Ví Dụ Sử Dụng Từng Script
\n+#### 1. **run_all.py** - Tự động toàn bộ (khuyên dùng)\n+```bash\n+# Cơ bản (dùng config tối ưu)\n+python run_all.py\n+\n+# Nâng cao\n+python run_all.py --device gpu --data-dir data_aligned --output-dir models\n+```\n+\n+#### 2. **train_cli.py** - Linh hoạt cho advanced users\n+```bash\n+# Mode all: đầy đủ pipeline\n+python train_cli.py --mode all --epochs 28\n+\n+# Mode single: 1 model\n+python train_cli.py --mode single --model resnet50 --epochs 28 --batch-size 16\n+\n+# Mode optimize: multi-round tuning\n+python train_cli.py --mode optimize --rounds 3 --epochs 12\n+```\n+\n+#### 3. **review_terminal.py** - Tuning interactively\n+```bash\n+python review_terminal.py\n+# Sau đó follow các gợi ý được in ra\n+```\n+\n+#### 4. **run_gui.py** - GUI Desktop\n+```bash\n+python run_gui.py\n+# Cửa sổ Tkinter sẽ mở ra, chọn 2 images để so sánh\n+```

## ⚙️ Cách Hoạt Động Chi Tiết

### Model Architecture Details

#### 1. ResNet50 - Residual Networks

```
INPUT: [1, 3, 224, 224]
  ↓
Convolutional Stem:
  ├─ Conv 7×7, stride=2 → [1, 64, 112, 112]
  ├─ BatchNorm + ReLU
  └─ MaxPool 3×3, stride=2 → [1, 64, 56, 56]
  ↓
Residual Blocks (4 groups):
  ├─ Layer 1: 3 blocks (64 → 64 channels) → [1, 64, 56, 56]
  ├─ Layer 2: 4 blocks (64 → 128 channels) → [1, 128, 28, 28]
  ├─ Layer 3: 6 blocks (128 → 256 channels) → [1, 256, 14, 14]
  └─ Layer 4: 3 blocks (256 → 512 channels) → [1, 512, 7, 7]
  ↓
Global Average Pooling: → [1, 512]
  ↓
Classification Head:
  ├─ FC 512 → 256 (ReLU)
  ├─ Dropout (0.5)
  └─ FC 256 → 8 (classes)
  ↓
OUTPUT: [1, 8] (logits)

Total parameters: ~90M
Pretrained on: ImageNet-1K
```

#### 2. EfficientNet-B0 - Compound Scaling

```
INPUT: [1, 3, 224, 224]
  ↓
Stem:
  ├─ Conv 3×3, stride=2 → [1, 32, 112, 112]
  └─ BatchNorm + ReLU
  ↓
MBConv Blocks (7 stages):
  ├─ Stage 1: 1× MBConv1 (32→16 channels)
  ├─ Stage 2: 2× MBConv6 (16→24 channels)
  ├─ Stage 3: 2× MBConv6 (24→40 channels)
  ├─ Stage 4: 3× MBConv6 (40→80 channels)
  ├─ Stage 5: 3× MBConv6 (80→112 channels)
  ├─ Stage 6: 4× MBConv6 (112→192 channels)
  └─ Stage 7: 1× MBConv6 (192→320 channels)
  ↓
Head:
  ├─ Conv 1×1 → [1, 1280]
  ├─ Global Average Pooling → [1, 1280]
  └─ FC 1280 → 8 (classes)
  ↓
OUTPUT: [1, 8] (logits)

Total parameters: ~7M (11x nhỏ hơn ResNet)
Pretrained on: ImageNet-1K
Advantage: Lightweight, fast inference (~10ms)
```

#### 3. Vision Transformer (ViT-B/16) - Attention-based

```
INPUT: [1, 3, 224, 224]
  ↓
Patch Embedding:
  ├─ Split image → 16×16 patches (14×14=196 patches)
  ├─ Flatten each patch: 3×16×16=768 dimensions
  ├─ Linear projection → [1, 196, 768]
  └─ Add positional encoding
  ↓
Transformer Encoder (12 layers):
  ├─ Multi-Head Self-Attention (12 heads, 64 dim each)
  │  ├─ Query, Key, Value projection
  │  ├─ Attention weights: A = softmax(Q*K^T / √d)
  │  └─ Output: A*V
  ├─ Feed-Forward Network (MLP)
  │  ├─ FC 768 → 3072 (GeLU)
  │  └─ FC 3072 → 768
  ├─ LayerNorm + Residual connections
  └─ Repeat 12 times
  ↓
Classification Head:
  ├─ CLS token → [1, 768]
  ├─ LayerNorm
  └─ FC 768 → 8 (classes)
  ↓
OUTPUT: [1, 8] (logits)

Total parameters: ~86M
Pretrained on: ImageNet-21K → ImageNet-1K
Advantage: Global context (attention xem toàn bộ ảnh)
```

---

### Loss Function & Optimization Details

#### Loss Function Stack

```python
# 1. CrossEntropyLoss (Base Loss)
CE_loss = -∑ y_i * log(softmax(logits)_i)

# 2. + Label Smoothing
y_smooth = (1 - epsilon) * y_hard + epsilon / num_classes
# Ví dụ: y_hard=[0,0,1,0,0,0,0,0] (hard)
#        y_smooth=[0.0187, ..., 0.9062, ..., 0.0187]  (soft)
# Lợi ích: Model không over-confident, prevent overfitting

# 3. + Mixup Augmentation (Batch-level)
λ ~ Beta(alpha, alpha)
mixed_x = λ*x_i + (1-λ)*x_j
mixed_y = λ*y_i + (1-λ)*y_j
# Lợi ích: Smooth decision boundary, data augmentation

# Final Loss
Total_Loss = CE(logits, mixed_y_smooth) + weight_decay * L2_norm(weights)
```

#### Optimizer: AdamW

```python
# Adam with Weight Decay
momentum_1 = β1 * momentum_1 + (1-β1) * gradient
momentum_2 = β2 * momentum_2 + (1-β2) * gradient^2
weight_update = -lr * momentum_1 / (√momentum_2 + eps)

# Weight Decay (L2 regularization)
weight = weight * (1 - weight_decay * lr)

# Hyperparameters
beta_1 = 0.9      (Momentum for 1st moment)
beta_2 = 0.999    (Momentum for 2nd moment)
eps = 1e-8
weight_decay = 1e-3  (L2 regularization)
```

#### Learning Rate Scheduler: ReduceLROnPlateau

```
Epoch 1-3: Warmup
  └─ lr increases linearly: 0 → base_lr

Epoch 4+: Standard training
  ├─ if val_loss not improve for patience=6 epochs:
  │  └─ lr *= decay_factor (default 0.5)
  └─ Repeat until LR reaches min_lr or max_epochs

Effect:
  - Early epochs: high LR (quick learning)
  - Late epochs: low LR (fine-tuning)
  - If stuck: reduce LR to escape local minima
```

---

### Early Stopping Mechanism

```python
# Monitor both:
1. Validation Accuracy Plateau
   if epochs_without_improvement >= patience:
       STOP

2. Train/Val Gap Threshold
   gap = train_acc - val_acc
   if gap > max_train_val_gap (0.12-0.16):
       STOP  # Prevent overfitting

# Benefits:
- Save training time (28 epochs → 5 epochs in practice)
- Better generalization (stop before overfitting)
```

---

## Hyperparameter Mặc Định Hiện Tại

Cấu hình này lấy trực tiếp từ optimal_configs.py:

| Model | lr | weight_decay | label_smoothing | mixup_alpha | epochs | early_stop_patience |
|---|---:|---:|---:|---:|---:|---:|
| ResNet50 | 6e-5 | 1.2e-3 | 0.16 | 0.35 | 28 | 6 |
| EfficientNet-B0 | 7e-5 | 1e-3 | 0.15 | 0.33 | 28 | 6 |
| ViT-B/16 | 5e-5 | 1.4e-3 | 0.20 | 0.42 | 32 | 7 |

**Giải thích từng tham số:**

- **lr** (Learning Rate): Tốc độ cập nhật trọng số
  - Thấp (5e-5): Fine-tuning, ổn định nhưng chậm
  - Cao (1e-3): Nhanh nhưng dễ overshoot
  
- **weight_decay**: L2 regularization strength (ngăn overfitting)
  - 0: Không regularization
  - 1e-3: Moderate (đang dùng)
  - 1e-2: Strong, có thể underfitting
  
- **label_smoothing**: Mềm hóa hard labels
  - 0: Hard labels [0,0,1,0] (dễ overfit)
  - 0.15-0.20: Soft labels (tối ưu cho medical)
  
- **mixup_alpha**: Blending strength
  - 0: Không augmentation
  - 0.33-0.42: Moderate mixing
  
- **epochs**: Maximum training rounds
  - 28-32: Đủ cho small dataset
  
- **early_stop_patience**: Bao nhiêu epoch không cải thiện mới dừng
  - 6-7: Balanced (dừng sớm, tránh overfit)

## File Quan Trọng Để Nộp Đồ Án
\n+---\n+\n+## 🧪 Kiểm Thử Nhanh\n+\n+### Unit Tests\n+```bash\n+python -m pytest tests/ -q\n+```\n+\n+**Gì được test:**\n+- Data loading (PillImageDataset)\n+- Inference utils (model loading, predictions)\n+- Metadata parsing (CSV)\n+\n+### Functional Test (First Time Setup)\n+```bash\n+# Test trên 1 batch nhỏ (2 epochs, dev mode)\n+python train_cli.py --mode single --model resnet50 --epochs 2 --batch-size 4\n+```\n+\n+---"

## Lỗi Thường Gặp Và Cách Xử Lý
## ⚠️ Lỗi Thường Gặp Và Cách Xử Lý
\n+### 1. ❌ Lỗi: "Cannot find data directory"
\n+**Nguyên nhân:** Chưa có thư mục dữ liệu hoặc sai vị trí.\n+\n+**Cách sửa:**\n+```bash\n+# Kiểm tra xem có thư mục data hoặc data_aligned không\n+ls -la | grep data\n+\n+# Nếu không có, hãy tạo structure\n+mkdir -p data_aligned/train data_aligned/val data_aligned/test\n+mkdir -p data_aligned/train/class_0 data_aligned/train/class_1  # etc.\n+\n+# Copy ảnh vào folder\n+cp ảnh_thuốc_1.jpg data_aligned/train/class_0/\n+\n+# Chạy lại\n+python run_all.py --data-dir data_aligned\n+```
\n+---
\n+### 2. ❌ Lỗi: "CUDA out of memory" (GPU)
\n+**Nguyên nhân:** Batch size quá lớn cho GPU bạn.\n+\n+**Cách sửa:**\n+```bash\n+# Cách 1: Dùng batch size nhỏ hơn\n+python train_cli.py --mode single --model resnet50 --batch-size 8\n+\n+# Cách 2: Dùng CPU thay vì GPU\n+python run_all.py --device cpu\n+\n+# Cách 3: Dùng model nhỏ (EfficientNet-B0)\n+python train_cli.py --mode single --model efficientnet_b0\n+```
\n+---
\n+### 3. ❌ Lỗi: "ModuleNotFoundError: No module named 'torch'"
\n+**Nguyên nhân:** PyTorch chưa cài đặt.\n+\n+**Cách sửa:**\n+```bash\n+# Cài lại requirements\n+pip install --upgrade pip\n+pip install -r requirements.txt\n+\n+# Nếu vẫn lỗi, cài PyTorch trực tiếp\n+# CPU:\n+pip install torch torchvision\n+\n+# GPU (CUDA 11.8):\n+pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118\n+```
\n+---
\n+### 4. ❌ Accuracy thấp (< 30%)
\n+**Nguyên nhân:** Dataset quá nhỏ, imbalanced, hoặc hyperparameter không tối ưu.\n+\n+**Cách sửa (theo thứ tự ưu tiên):**\n+\n+**a) Tăng kích cỡ dataset** (quan trọng nhất)\n+```bash\n+# Tây thêm ảnh vào train/ folder\n+# Lý tưởng: 50-100 ảnh/class\n+# Hiện tại: ~20 ảnh/class\n+```\n+\n+**b) Điều chỉnh hyperparameter**\n+```bash\n+# Cách 1: Dùng review_terminal để xem curves\n+python review_terminal.py\n+# Bạn sẽ thấy gợi ý (tăng/giảm LR, etc.)\n+\n+# Cách 2: Multi-round optimization\n+python train_cli.py --mode optimize --rounds 3 --epochs 20\n+\n+# Cách 3: Manual tuning\n+python train_cli.py --mode single --model resnet50 \\\n+  --learning-rate 5e-5 --weight-decay 2e-3 --label-smoothing 0.2\n+```
\n+**c) Train với kích thước ảnh cao hơn** (advanced)\n+```python\n+# Sửa trong feature.py, tìm dòng:\nIMAGE_SIZE = 224\n+# Đổi thành:\nIMAGE_SIZE = 384  # Hoặc 512\n+# (Nhưng sẽ chậm hơn + cần RAM hơn)\n+```
\n+---
\n+### 5. ❌ Lỗi: "class_to_idx mismatch" (Inference)\n+\n+**Nguyên nhân:** Checkpoint được save từ dataset khác (số classes khác).\n+\n+**Cách sửa:**\n+```bash\n+# Train lại từ scratch\n+python run_all.py\n+# Checkpoint mới sẽ có đúng class_to_idx\n+```
\n+---
\n+### 6. ❌ Training quá chậm\n+\n+**Nguyên nhân:** Dùng CPU, hoặc GPU không được detect.\n+\n+**Cách sửa:**\n+```bash\n+# Kiểm tra GPU\n+python -c \"import torch; print(torch.cuda.is_available())\"\n+# Nếu output: False → GPU không sẵn sàng\n+\n+# Install GPU drivers (Windows)\n+# 1. Download NVIDIA driver từ nvidia.com\n+# 2. Install driver\n+# 3. Install CUDA Toolkit 11.8\n+# 4. Re-install PyTorch:\n+pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118\n+\n+# Kiểm tra lại\n+python -c \"import torch; print(torch.cuda.is_available())\"\n+```
\n+---

### 7. ❌ Git push lỗi: "File too large"
\n+**Nguyên nhân:** Checkpoint .pt (100-200MB) quá to.\n+\n+**Cách sửa:**\n+```bash\n+# Thêm .pt files vào .gitignore\n+echo "*.pt" >> .gitignore\n+echo "models/" >> .gitignore\n+\n+# Xóa khỏi git history\n+git rm --cached models/*.pt\n+\n+# Commit\n+git add .gitignore\n+git commit -m \"Ignore large checkpoint files\"\n+git push\n+```
\n+---
\n+### 8. ❌ Lỗi: \"RuntimeError: Expected all tensors on same device\"\n+\n+**Nguyên nhân:** Model trên GPU nhưng data trên CPU (hoặc ngược lại).\n+\n+**Cách sửa:**\n+Code thường auto-detect, nhưng nếu lỗi:\n+```bash\n+# Chạy với CPU\n+python run_all.py --device cpu\n+\n+# Hoặc fix trong code (advanced):\n+# Tìm trong train.py, models.py\n+# model = model.to(device)\n+# x = x.to(device)\n+```
\n+---

## ✅ Checklist Trước Khi Nộp
\n+### Data & Environment\n+- [ ] Có thư mục `data_aligned/` hoặc `data/` với train/val/test\n+- [ ] Ít nhất 20-30 ảnh/class (tốt nhất 50+)\n+- [ ] Đã chạy `pip install -r requirements.txt` thành công\n+- [ ] `python -c \"import torch; print(torch.cuda.is_available())\"` in được GPU status\n+\n+### Training & Evaluation\n+- [ ] Đã chạy `python run_all.py` ít nhất 1 lần\n+- [ ] Không có lỗi trong terminal\n+- [ ] Models/ folder có 9 files (.pt + .metrics.json + .history.json)\n+- [ ] models/ folder có training_curves.png (3 files)\n+- [ ] models/reports/latest/ có confusion_matrix_*.png (4 files)\n+\n+### Results & Output\n+- [ ] models/evaluation_summary.csv có data (4 dòng: 3 models + ensemble)\n+- [ ] Accuracy > 20% (hoặc ít nhất > random 12.5% cho 8 classes)\n+- [ ] Macro-F1 > 0.10\n+- [ ] models/evaluation_comparison.png là hình bar chart rõ ràng\n+\n+### Submission Files\n+- [ ] **Checkpoints:** 3 file `*_epillid_best.pt` (100% cần)\n+- [ ] **Metrics:** `models/evaluation_summary.csv` + JSON\n+- [ ] **Visualizations:** `*_training_curves.png` + confusion matrices\n+- [ ] **Source code:** src/ folder đầy đủ\n+- [ ] **This README:** README.md + requirements.txt\n+\n+### Code Quality\n+- [ ] `python -m pytest tests/ -q` chạy hết test cases\n+- [ ] Không có uncommitted changes: `git status` sạch\n+- [ ] Đã `git push` lên GitHub\n+- [ ] Repo có README + LICENSE + .gitignore\n+\n+### Colab (Nếu dùng)\n+- [ ] THUOC_Colab_Train_Evaluate.ipynb có 9 cells\n+- [ ] Đã test notebook từ trên Colab (chạy hết cells)\n+- [ ] Output ZIP có đầy đủ artifacts\n+\n+---\n+\n+## 📋 Cấu Trúc Thư Mục (Chi Tiết)\n+\n+```\n+THUOC/\n+\u251c\u2500\u2500 \ud83d\udcscript files\n+\u2502   \u251c\u2500 run_all.py (\ud83d\ude80 Main)\n+\u2502   \u251c\u2500 train_cli.py (Advanced)\n+\u2502   \u251c\u2500 run_gui.py (GUI)\n+\u2502   \u251c\u2500 review_terminal.py (Tuning)\n+\u2502   \u251c\u2500 optimal_configs.py (Hyperparams)\n+\u2502   \u2514\u2500 THUOC_Colab_Train_Evaluate.ipynb (\ud83e\udddc Colab)\n+\u2502\n+\u251c\u2500\u2500 \ud83d\udce6 src/ (Core modules)\n+\u2502   \u251c\u2500 train.py (\u2699\ufe0f Training engine)\n+\u2502   \u251c\u2500 models.py (\ud83e\udde0 Model builders)\n+\u2502   \u251c\u2500 features.py (\ud83d\udccaData pipeline)\n+\u2502   \u251c\u2500 pipeline.py (\ud83d\udd04 Orchestration)\n+\u2502   \u251c\u2500 inference.py (\ud83c\udfaf Predictions)\n+\u2502   \u251c\u2500 evaluate_report.py (\ud83d\udcc8 Metrics)\n+\u2502   \u251c\u2500 build_epillid_data.py (\ud83d\udd27 Data integration)\n+\u2502   \u251c\u2500 metadata.py (\ud83c\udff7\ufe0f Metadata)\n+\u2502   \u251c\u2500 self_learning.py (\ud83d\udcda Feedback)\n+\u2502   \u2514\u2500 gui_tk.py (\ud83d\udda5\ufe0f GUI)\n+\u2502\n+\u251c\u2500\u2500 \ud83d\udcc2 DATA FOLDERS\n+\u2502   \u251c\u2500 data_aligned/ (train/val/test)\n+\u2502   \u251c\u2500 data/ (backup data)\n+\u2502   \u251c\u2500 demo_images/ (sample images)\n+\u2502   \u2514\u2500 Medicine_Details_Deeplearning.csv\n+\u2502\n+\u251c\u2500\u2500 \ud83d\udcbe models/ (OUTPUT folder)\n+\u2502   \u251c\u2500 *.pt (3 checkpoints)\n+\u2502   \u251c\u2500 *.metrics.json (3 files)\n+\u2502   \u251c\u2500 *.history.json (3 files)\n+\u2502   \u251c\u2500 *training_curves.png (3 files)\n+\u2502   \u251c\u2500 evaluation_summary.csv (Main results)\n+\u2502   \u251c\u2500 evaluation_comparison.png (Bar chart)\n+\u2502   \u2514\u2500 reports/latest/ (Detailed reports)\n+\u2502\n+\u251c\u2500\u2500 \ud83e\uddea tests/ (Unit tests)\n+\u2502   \u251c\u2500 test_features.py\n+\u2502   \u251c\u2500 test_inference_utils.py\n+\u2502   \u2514\u2500 test_metadata.py\n+\u2502\n+\u2514\u2500\u2500 \u2699\ufe0f CONFIG FILES\n    \u251c\u2500 requirements.txt (\ud83d\udccc Documented packages)\n    \u251c\u2500 README.md (This file)\n    \u251c\u2500 .gitignore\n    \u2514\u2500 .git/ (Git history)\n+```
\n+---\n+\n+## \ud83d\udcc6 Tóm Tắt Tính Năng Chính\n+\n+| Tính Năng | \u0110ặc Điểm |\n+|---|---|\n+| **3 Model Architectures** | ResNet50 (90M), EfficientNet-B0 (7M), ViT-B/16 (86M) |\n+| **Training Strategy** | 2-stage (frozen backbone → full training) |\n+| **Regularization** | Label smoothing + Mixup + Weight decay |\n+| **Optimization** | AdamW + ReduceLROnPlateau + Early stopping |\n+| **Data Augmentation** | ColorJitter, Flip, Rotation, Normalization, Mixup |\n+| **Ensemble** | Weighted 3-model voting |\n+| **Evaluation Metrics** | Accuracy, Macro-F1, Confusion Matrix |\n+| **Inference** | Single image prediction + confidence scoring |\n+| **GPU Support** | CUDA/CPU auto-detect |\n+| **Colab Ready** | Jupyter notebook với 9 cells |\n+| **GUI Application** | Tkinter desktop app for image comparison |\n+| **Hyperparameter Tuning** | Multi-round optimization mode |\n+\n+---\n+\n+## \ud83d\udd10 Giấy Phép & Citation\n+\n+Nếu bạn dùng project này, vui lòng cite:\n+\n+```bibtex\n+@inproceedings{THUOC2024,\n+  title={THUOC: Medicine Pill Classification using Deep Learning},\n+  author={Your Name},\n+  year={2024},\n+  note={University Project}\n+}\n+```\n+\n+---\n+\n+## \ud83d\udc4b Support & Contribution\n+\n+**Có vấn đề?**\n+1. Check phần [⚠️ Lỗi Thường Gặp](#%EF%B8%8F-l%E1%BB%97i-th%C6%B0%E1%BB%9Dng-g%E1%BA%B7p-v%C3%A0-c%C3%A1ch-x%E1%BB%AD-l%C3%BD)\n+2. Read terminal output (lỗi thường chi tiết)\n+3. Check internet connection (download pretrained models cần network)\n+\n+**Muốn contribute?**\n+1. Fork repo\n+2. Tạo branch mới: `git checkout -b feature/improvement`\n+3 Commit changes: `git commit -m \"Add feature\"`\n+4. Push & create Pull Request\n+\n+---\n+\n+## 📞 Contact\n+\n+- **Issues:** GitHub Issues tab\n+- **Discussions:** GitHub Discussions\n+- **Email:** your-email@example.com\n+\n+---\n+\n+**Last Updated:** March 2024  \n+**Version:** 2.0 (Comprehensive documentation)\n+**Status:** ✅ Production-ready"

3. **Accuracy thấp (dưới 50%):**
Dataset quá nhỏ hoặc class imbalance. Hãy:
- Tăng kích cỡ dataset.
- Điều chỉnh hyperparameter: giảm lr, tăng epochs.
- Chạy lại với: `python run_all.py --data-dir data_aligned`

4. **Muốn nộp đồ án nhưng thiếu report:**
Chạy lại lệnh all-in-one để hệ thống sinh đủ file.

```bash
python run_all.py
```

## Checklist Trước Khi Nộp

1. Đã có đủ 3 checkpoint trong models.
2. Đã có evaluation_summary.csv và evaluation_comparison.png.
3. Đã có confusion matrix trong models/reports/latest.
4. Đã có training curves cho cả 3 model.
5. README này đi kèm source code.
6. Kiểm tra git status: `git status` (không nên có uncommitted changes).