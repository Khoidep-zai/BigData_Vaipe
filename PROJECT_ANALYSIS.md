# THUOC Project - Phân Tích Cấu Trúc Chi Tiết

## 1. FILE MAPPING - CHỨC NĂNG CỦA TỪNG FILE

### Root Files (Files chính trong thư mục gốc)

| File | Chức Năng |
|---|---|
| **run_all.py** | 🚀 Main entry point - Chạy toàn bộ pipeline (train 3 models + evaluate + show results) |
| **train_cli.py** | CLI cho training - Hỗ trợ 3 mode: `all` (pipeline), `single` (1 model), `optimize` (tuning rounds) |
| **run_gui.py** | Launcher cho GUI Tkinter - Khởi động ứng dụng desktop phân loại thuốc |
| **review_terminal.py** | Công cụ interactive tuning - Review training curves + suggest hyperparams changes in real-time |
| **optimal_configs.py** | Hyperparameter configs được tối ưu cho 3 models (OPTIMAL_CONFIGS + TUNING_CANDIDATES) |
| **requirements.txt** | Dependencies list (torch, torchvision, numpy, PIL, scikit-learn, etc.) |
| **README.md** | Hướng dẫn sử dụng nhanh cho người dùng |

### src/ Folder - Core Modules

| File | Chức Năng |
|---|---|
| **train.py** | ⚙️ **Training engine** - Trainloop chính: loss, optimizer (AdamW), scheduler (ReduceLROnPlateau), mixup, label smoothing |
| **models.py** | 🧠 **Model builders** - Create & load 3 architectures: ResNet50, EfficientNet-B0, ViT-B/16 từ torchvision |
| **features.py** | 📊 **Data transforms & Dataset** - PillImageDataset class, augmentation pipeline (flip, rotation, color jitter), image preprocessing |
| **pipeline.py** | 🔄 **Orchestration** - Main pipeline logic: data discovery, train all 3 models, evaluate, generate reports & visualizations |
| **inference.py** | 🎯 **Prediction engine** - Model inference, cache management, confidence scoring, pixel-level feature comparison |
| **evaluate_report.py** | 📈 **Metrics & reporting** - Evaluate single model, compute Accuracy/F1, confusion matrix, CSV export |
| **build_epillid_data.py** | 🔧 **Data integration** - Build dataset từ ePillID source (copy/hardlink/symlink files vào train/val/test splits) |
| **metadata.py** | 🏷️ **Metadata indexing** - Parse CSV metadata, match class names với medicine properties (composition, dosage, color, shape) |
| **self_learning.py** | 📚 **Feedback & hard examples** - Log user feedback, collect hard cases để fine-tune model |
| **gui_tk.py** | 🖥️ **GUI application** - Tkinter app: compare 2 ảnh, show scores (similarity, color, size, texture), display metadata |
| **__init__.py** | Package initialization |

## 2. DATA PIPELINE - QUY TRÌNH XỬ LÝ DỮ LIỆU

### 📂 Data Structure
```
data_aligned/  hoặc  data/
├── train/
│   ├── class_0/          (e.g., "cefadroxil_500mg_0.5g")
│   │   ├── img001.jpg
│   │   ├── img002.png
│   │   └── ...
│   ├── class_1/
│   │   └── ...
│   └── class_N/
├── val/
│   ├── class_0/
│   ├── class_1/
│   └── ...
└── test/
    ├── cefadroxil_500mg_0.5g/
    ├── golddicron_30mg/
    ├── kavasdin_5_5mg/
    ├── panactol_500mg/
    ├── sergurop_10mg/
    ├── thuoc_chua_dinh_danh_0/
    ├── thuoc_chua_dinh_danh_1/
    └── thuoc_ngoai_don_class_107/
```

### 🔄 Data Pipeline Steps

#### **Step 1: Data Discovery** (`pipeline.py` → `discover_data_dir()`)
- Auto-detect data directory (prefer `data_aligned` if richer, fallback to `data`)
- Validate structure: train/val/test subfolders tồn tại
- Extract class names từ folder names

#### **Step 2: Dataset Loading** (`features.py` → `PillImageDataset`)
```
Files → PillImageDataset
  ├─ Scan thư mục train/val/test
  ├─ Find all images (.jpg, .jpeg, .png)
  ├─ Build class_to_idx mapping (e.g., {"cefadroxil_500mg_0.5g": 0})
  └─ Create ImageSample (path, label) tuples
```

#### **Step 3: Image Augmentation** (Training)
```
Raw Image → PilImage
  │
  ├─ pil_loader: Convert to RGB
  ├─ focus_on_object: Center crop 85% để focus vào viên thuốc
  │   └─ (loại bỏ background)
  │
  ├─ [TRAINING TRANSFORMS]
  │  ├─ Resize to 224×224
  │  ├─ ColorJitter (brightness, contrast, saturation ±8%)
  │  ├─ RandomHorizontalFlip (50% chance)
  │  ├─ RandomRotation (±5°)
  │  ├─ ToTensor
  │  └─ Normalize (ImageNet mean/std)
  │
  └─ [INFERENCE TRANSFORMS] (No aug)
     ├─ Resize to 224×224
     ├─ ToTensor
     └─ Normalize (ImageNet mean/std)
```

**Normalization Constants:**
```python
mean = [0.485, 0.456, 0.406]  # ImageNet RGB mean
std  = [0.229, 0.224, 0.225]  # ImageNet RGB std
```

#### **Step 4: DataLoader Creation** (`train.py` → `create_dataloaders()`)
```
Train Dataset (transform=train) → DataLoader
  ├─ Batch size: 16 (adaptive, min to dataset size)
  ├─ Shuffle: True (với seed=42 cho reproducibility)
  ├─ Num workers: 0 (Windows default, 2 on Linux/Mac)
  └─ Pin memory: True (nếu GPU available)

Validation Dataset (transform=eval) → DataLoader
  ├─ Batch size: min(16, dataset_size)
  ├─ Shuffle: False
  └─ Pin memory: True (nếu GPU)

Train Metric Loader (clean, no aug) → DataLoader
  └─ Dùng để compute train accuracy trên original samples
```

**Stratified Holdout Logic:**
- Nếu val set quá nhỏ (<24 samples): lấy 15% từ train set as additional val
- Stratified by class: mỗi class giữ tối thiểu 1 sample

#### **Mixup Data Augmentation** (Training)
```
For each batch:
  λ ~ Beta(α, α)  [α = 0.10 to 0.42 depending on model]
  
  mixed_x = λ × x_i + (1-λ) × x_j  (blend pairs of images)
  Loss = λ × L(pred, y_i) + (1-λ) × L(pred, y_j)
```

## 3. TRAINING ALGORITHM OVERVIEW - QUY TRÌNH HUẤN LUYỆN

### 🔧 Training Configuration by Model

```python
OPTIMAL_CONFIGS = {
    "resnet50": {
        "lr": 2e-4,                    # Learning rate
        "weight_decay": 5e-4,          # L2 regularization
        "label_smoothing": 0.06,       # CrossEntropy smoothing
        "mixup_alpha": 0.10,           # Mixup strength
        "epochs": 36,                  # Max epochs
        "batch_size": 16,              # Batch size
        "early_stop_patience": 8,      # Wait 8 epochs no improve
        "max_train_val_gap": 0.16,     # Stop if train_acc - val_acc > 0.16
        "freeze_backbone_epochs": 2,   # Freeze layers 1-2 epochs
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
        "lr": 5e-5,                    # Lower LR for ViT
        "weight_decay": 1.4e-3,        # Stronger regularization
        "label_smoothing": 0.20,       # Stronger smoothing
        "mixup_alpha": 0.42,           # Stronger mixup
        "epochs": 32,
        "batch_size": 16,
        "early_stop_patience": 7,
        "max_train_val_gap": 0.12,     # Tighter gap for ViT
        "freeze_backbone_epochs": 5,   # Freeze backbone longer (more stable)
    }
}
```

### 📊 Training Loop (`train.py` → `train()`)

#### **Initialization Phase**
```
1. Set random seeds (reproducibility)
   ├─ np.random.seed(42)
   ├─ torch.manual_seed(42)
   └─ torch.cuda.manual_seed_all(42)

2. Create DataLoaders (train, val, train_metric)

3. Build model from pretrained weights
   ├─ ResNet50(ImageNet1K_V2) → Replace fc layer
   ├─ EfficientNet-B0(ImageNet1K_V1) → Replace classifier
   └─ ViT-B/16(ImageNet1K_V1) → Replace heads.head

4. Setup Loss & Optimizer
   ├─ Loss: CrossEntropyLoss(label_smoothing=LS)
   ├─ Optimizer: AdamW(lr, weight_decay)
   └─ Scheduler: ReduceLROnPlateau (reduce LR if val loss plateau)

5. Other setups
   ├─ Gradient scaler for mixed precision (CUDA)
   ├─ Warmup schedule: 3 epochs linear warmup
   └─ Early stopping counter
```

#### **Per-Epoch Training**
```
for epoch in 1..max_epochs:

  # ========== BACKBONE FREEZING LOGIC ==========
  if epoch <= freeze_backbone_epochs:
    Freeze all backbone layers (conv, encoder, etc)
    Only train classifier/head
  else:
    Unfreeze everything
  
  # ========== WARMUP LEARNING RATE ==========
  if epoch <= 3:
    warmup_lr = base_lr × (0.4 + 0.6 × epoch/3)
    Set optimizer.param_groups[0]['lr'] = warmup_lr
  
  # ========== TRAINING LOOP ==========
  model.train()
  for batch in train_loader:
    images, labels, paths = batch  # Shape: (B, 3, 224, 224)
    
    # Mixup augmentation
    mixed_images, labels_a, labels_b, lam = mixup_batch(
      images, labels, alpha=mixup_alpha, device
    )
    
    # Forward pass with AMP (Automatic Mixed Precision)
    optimizer.zero_grad()
    with torch.amp.autocast("cuda", enabled=use_amp):
      logits = model(mixed_images)  # (B, num_classes)
      loss = lam * loss_fn(logits, labels_a) + (1-lam) * loss_fn(logits, labels_b)
    
    # Backward pass
    scaler.scale(loss).backward()
    
    # Gradient clipping
    if grad_clip_norm > 0:
      scaler.unscale_(optimizer)
      torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    
    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
  
  # ========== EVALUATION ON TRAIN & VAL ==========
  train_acc, train_loss = evaluate(model, train_metric_loader, device)
  val_acc, val_loss = evaluate(model, val_loader, device)
  
  # ========== LEARNING RATE SCHEDULING ==========
  if epoch > warmup_epochs:
    scheduler.step(val_loss)  # ReduceLROnPlateau
  
  # ========== EARLY STOPPING LOGIC ==========
  gap = train_acc - val_acc
  
  # Guard 1: Train/Val gap too large
  if gap > max_train_val_gap and epoch >= gap_guard_start_epoch:
    epochs_with_large_gap += 1
    if epochs_with_large_gap >= max_large_gap_epochs:
      STOP (avoid divergence)
  
  # Guard 2: No improvement patience
  if val_acc > best_acc:
    best_acc = val_acc
    Save checkpoint to models/{model}_epillid_best.pt
    epochs_without_improve = 0
  else:
    epochs_without_improve += 1
    if epochs_without_improve >= early_stop_patience:
      STOP
  
  # ========== LOG & DISPLAY ==========
  print(f"Epoch {epoch:5} | TrLoss {train_loss:7.4f} | VaLoss {val_loss:7.4f} | TrAcc {train_acc:6.4f} | VaAcc {val_acc:6.4f}")
```

#### **Checkpoint Saving**
```
When val_acc improves:
  torch.save({
    "model_state_dict": model.state_dict(),
    "num_classes": num_classes,
    "class_to_idx": class_to_idx,  # IMPORTANT: Save class mapping
    "val_acc": val_acc,
  }, "models/{model}_epillid_best.pt")

Also save metrics JSON:
  {
    "best_val_acc": 0.75,
    "epochs": 12,
    "model": "resnet50",
    "lr": 0.0002,
    "weight_decay": 0.0005,
    "label_smoothing": 0.06,
    "mixup_alpha": 0.10
  }

And training history:
  {
    "epoch": [1, 2, ..., 12],
    "train_loss": [...],
    "val_loss": [...],
    "train_acc": [...],
    "val_acc": [...]
  }
```

#### **Training Curve Visualization**
- Plot train_loss vs val_loss
- Plot train_acc vs val_acc
- Apply EMA smoothing (α=0.35) to raw curves
- Highlight best epoch (minimum val_loss)
- Show train/val gap as shaded region

## 4. MODEL ARCHITECTURES

### ResNet50
```
Input: (B, 3, 224, 224)
  │
  └─ Pre-trained ResNet50 backbone (ImageNet1K_V2)
     ├─ Conv1 + BatchNorm + ReLU
     ├─ Layer1, Layer2, Layer3, Layer4 (residual blocks)
     └─ Average pooling → (B, 2048)
  
     Replace: model.fc = Linear(2048, num_classes)
  
Output: (B, num_classes) logits
```

**Backbone freezing:**
- Freeze: conv1, bn1, layer1, layer2, layer3, layer4
- Train only: fc (classifier head)

### EfficientNet-B0
```
Input: (B, 3, 224, 224)
  │
  └─ Pre-trained EfficientNet-B0 (ImageNet1K_V1)
     ├─ Stem (Conv + BN + SiLU)
     ├─ MBConv blocks (mobile inverted bottleneck)
     ├─ Head (Conv + BN + SiLU)
     └─ Average pooling → (B, 1280)
  
     Replace: model.classifier[-1] = Linear(1280, num_classes)
  
Output: (B, num_classes) logits
```

**Backbone freezing:**
- Freeze: features (all MBConv layers)
- Train only: classifier

### Vision Transformer (ViT-B/16)
```
Input: (B, 3, 224, 224)
  │
  └─ Pre-trained ViT-B/16 (ImageNet1K_V1)
     ├─ Patch embedding (16×16 patches)
     ├─ Transformer encoder blocks
     ├─ Classification token [CLS]
     └─ (B, 768) hidden state
  
     Replace: model.heads.head = Linear(768, num_classes)
  
Output: (B, num_classes) logits
```

**Backbone freezing:**
- Freeze: encoder (all transformer blocks)
- Train only: heads.head

## 5. INFERENCE FLOW - QUY TRÌNH DỰ ĐOÁN

### Single Model Inference (`inference.py`)

```
Query Image → PIL Image
  │
  ├─ pil_loader(path): Load & convert to RGB
  ├─ focus_on_object(img, scale=0.85): Center crop
  │
  ├─ build_transforms(train=False): Apply eval transforms
  │  ├─ Resize(224, 224)
  │  ├─ ToTensor
  │  └─ Normalize (ImageNet)
  │
  └─ Model forward pass
     ├─ Load model from checkpoint (with cached class_to_idx)
     ├─ model.eval() + torch.no_grad()
     ├─ logits = model(image)  # (1, num_classes)
     │
     ├─ Predictions:
     │  ├─ probabilities = softmax(logits)
     │  ├─ predicted_class_idx = argmax(logits)
     │  ├─ predicted_class_name = class_idx_to_name[predicted_class_idx]
     │  └─ confidence = max(probabilities)
     │
     └─ Return: (predicted_class, confidence_score)
```

### Compare Two Images (`inference.py` → `compare_pill_images()`)

```
Sample Image (from database)
Query Image (user uploads)

For each image:
  ├─ Load & preprocess (224×224, RGB, normalized)
  ├─ Extract features from model backbone:
  │  └─ ResNet50: model.avgpool (before fc) → (2048,)
  │  └─ EfficientNet: model.head (before classifier) → (1280,)
  │  └─ ViT: transformer output → (768,)
  │
  ├─ Compute image statistics (for color/texture score)
  │  ├─ RGB mean color in focused pill area
  │  ├─ Image dimensions
  │  └─ Aspect ratio
  │
  └─ Generate pixel-level description (simplified texture)

Comparison Scores:
  ├─ similarity_score = cosine_distance(feature_sample, feature_query)
  ├─ color_score = 1 - RGB_distance(sample, query)
  ├─ size_score = aspect_ratio_similarity
  ├─ texture_score = gradient_pattern_similarity
  │
  └─ Overall prediction:
     ├─ IF similarity_score > sim_threshold AND color_score > color_threshold:
     │    predicted_class = sample's class
     ├─ ELSE:
     │    predicted_class = "No match found"
     │
     └─ Return ComparisonResult(
        predicted_class="cefadroxil_500mg_0.5g",
        similarity_score=0.82,
        color_score=0.75,
        size_score=0.88,
        texture_score=0.70,
        num_true_features=3,
        is_true=True,
        details={...}
     )
```

### Ensemble Predictions (Optional)

```
For each of 3 models:
  ├─ Load checkpoint
  ├─ Forward pass → probabilities
  └─ Add to ensemble

Ensemble voting:
  ├─ Average probabilities across models
  ├─ Take argmax → ensemble prediction
  │
  └─ Return ensemble result + individual model results
```

### Feature Caching (`inference.py` → `_MODEL_CACHE`)
```python
Cache key: (model_name, checkpoint_path, device)
Cache value: (loaded_model, idx_to_class_mapping)

Benefit: Avoid reloading model multiple times in same session
         (especially important for GUI app where user compares many images)
```

## 6. COMPLETE REQUIREMENTS & DEPENDENCIES

### requirements.txt - Exact Versions

```
torch>=2.0.0              # PyTorch deep learning framework
torchvision>=0.15.0       # Pre-trained models, transforms
numpy>=1.24.0             # Numerical computing
Pillow>=10.0.0            # Image processing (PIL)
scikit-learn>=1.3.0       # Metrics (accuracy_score, f1_score, confusion_matrix)
tqdm>=4.66.0              # Progress bars
matplotlib>=3.7.0         # Plotting (training curves, confusion matrices)
pandas>=2.0.0             # CSV reading for metadata
```

**Version breakdown:**
| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=2.0.0 | Core ML framework, tensor ops, autograd |
| torchvision | >=0.15.0 | Pretrained ResNet50, EfficientNet-B0, ViT-B/16 |
| numpy | >=1.24.0 | Array operations, mixup blending |
| Pillow | >=10.0.0 | Image loading/processing (focus_on_object, crop) |
| scikit-learn | >=1.3.0 | accuracy_score, f1_score, confusion_matrix, stratified split |
| tqdm | >=4.66.0 | Progress bars in training loops |
| matplotlib | >=3.7.0 | Plot training curves, confusion matrix heatmaps |
| pandas | >=2.0.0 | Read/write CSV (metadata, evaluation results) |

**Why no version caps?**
- `>=` allows newer versions with bug fixes
- Tested with listed versions, newer usually backward compatible
- GPU users might need CUDA-specific builds (installed separately)

## 7. OUTPUT ARTIFACTS - KẾT QUẢ SINH RA

### After Training Completes

```
models/
├─ resnet50_epillid_best.pt              (Checkpoint)
├─ resnet50_epillid_best.metrics.json    (Best metrics)
├─ resnet50_epillid_history.json         (All epoch metrics)
├─ resnet50_training_curves.png          (Loss/Acc plots)
│
├─ efficientnet_b0_epillid_best.pt
├─ efficientnet_b0_epillid_best.metrics.json
├─ efficientnet_b0_epillid_history.json
├─ efficientnet_b0_training_curves.png
│
├─ vit_b_16_epillid_best.pt
├─ vit_b_16_epillid_best.metrics.json
├─ vit_b_16_epillid_history.json
├─ vit_b_16_training_curves.png
│
├─ evaluation_summary.csv                (Accuracy, F1, num_samples for each model)
├─ evaluation_comparison.png             (Bar chart comparing models)
├─ training_results_table.csv            (Summary table)
├─ training_results_table.md             (Markdown table)
│
└─ reports/
   ├─ latest/
   │  ├─ evaluation_summary.json
   │  ├─ evaluation_summary.csv
   │  └─ tuning_summary.json
   │
   └─ smoke_final/
      ├─ evaluation_summary.json
      ├─ evaluation_summary.csv
      └─ tuning_summary.json
```

### Checkpoint File Structure (`.pt`)

```json
{
  "model_state_dict": {...},        // PyTorch model weights
  "num_classes": 8,                 // Number of output classes
  "class_to_idx": {
    "cefadroxil_500mg_0.5g": 0,
    "golddicron_30mg": 1,
    "kavasdin_5_5mg": 2,
    ...
  },
  "val_acc": 0.75                   // Validation accuracy at save time
}
```

### Metrics JSON Format

```json
{
  "best_val_acc": 0.75,
  "epochs": 12,
  "model": "resnet50",
  "lr": 0.0002,
  "weight_decay": 0.0005,
  "label_smoothing": 0.06,
  "mixup_alpha": 0.1
}
```

### Training History JSON Format

```json
{
  "epoch": [1, 2, 3, ..., 12],
  "train_loss": [0.523, 0.451, 0.398, ...],
  "val_loss": [0.612, 0.545, 0.532, ...],
  "train_acc": [0.75, 0.78, 0.81, ...],
  "val_acc": [0.72, 0.74, 0.75, ...]
}
```

### Evaluation Summary CSV

```csv
Model,Accuracy,Macro-F1,Best Epoch,Train/Val Gap
ResNet50,0.75,0.82,12,0.06
EfficientNet-B0,0.78,0.85,15,0.05
ViT-B/16,0.72,0.80,10,0.08
```

### Confusion Matrix Output
- Per-model confusion matrices saved as heatmaps (PNG)
- Ensemble confusion matrix for combined predictions
- JSON format (raw counts) for programmatic access

## 8. SUMMARY TABLE - QUICK REFERENCE

| Aspect | Details |
|--------|---------|
| **Models** | ResNet50, EfficientNet-B0, ViT-B/16 (pretrained ImageNet) |
| **Loss Function** | CrossEntropyLoss with label smoothing |
| **Optimizer** | AdamW (adaptive momentum, weight decay L2) |
| **Scheduler** | ReduceLROnPlateau (reduce LR on plateau) |
| **Data Aug (Train)** | Mixup, ColorJitter, RandomFlip, Rotation, Normalization |
| **Early Stopping** | Patience=8 or early gap guard (train-val divergence) |
| **Backbone Freezing** | First N epochs train only classifier, then unfreeze all |
| **Gradient Clipping** | Optional clip_grad_norm=1.0 for stability |
| **Warmup** | 3 epochs linear warmup to base_lr |
| **Mixed Precision** | torch.amp.autocast on CUDA for speed |
| **Batch Size** | 16 (adaptive if dataset small) |
| **Input Size** | 224×224 (standard for ImageNet models) |
| **Eval Metrics** | Accuracy, Macro-F1, Confusion Matrix |
| **Class Mapping** | Stored in checkpoint to avoid mismatch during inference |
| **Inference Cache** | Models cached in memory per session |

---

**Generated:** 2024-03-24
**Project:** THUOC - Pill Classification from Images
**Context:** Deep learning project for medicine identification using image classification
