# THUOC PROJECT - VISUAL ARCHITECTURE & DATA FLOW DIAGRAMS

## 1. SYSTEM ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      THUOC - Pill Classification System                 │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          ENTRY POINTS                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  python run_all.py          ──→ Auto-discovery + Train all 3 models    │
│  python train_cli.py        ──→ CLI mode (single model or optimize)    │
│  python run_gui.py          ──→ Desktop GUI app (Tkinter)              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
        ┌───────────────────────────────────────────────────┐
        │         DATA LOADING & DISCOVERY LAYER           │
        ├───────────────────────────────────────────────────┤
        │                                                   │
        │  discover_data_dir()  [pipeline.py]              │
        │    └─→ Find: data_aligned/ or data/              │
        │    └─→ Populate: {class_0, class_1, ..., N}      │
        │                                                   │
        │  PillImageDataset      [features.py]             │
        │    └─→ Scan train/val/test folders               │
        │    └─→ Load .jpg, .png files                      │
        │    └─→ Map classname → index                      │
        │                                                   │
        └───────────────────────────────────────────────────┘
                                    │
                                    ↓
        ┌───────────────────────────────────────────────────┐
        │      IMAGE PREPROCESSING & AUGMENTATION LAYER     │
        ├───────────────────────────────────────────────────┤
        │                                                   │
        │  pil_loader()                                     │
        │    └─→ Load image file as RGB PIL Image           │
        │                                                   │
        │  focus_on_object()                                │
        │    └─→ Center crop to 85% (remove background)     │
        │                                                   │
        │  build_transforms()  [Training vs. Eval]          │
        │    ├─ TRAIN: Resize → ColorJitter → Flip →       │
        │    │         Rotate → ToTensor → Normalize        │
        │    └─ EVAL:  Resize → ToTensor → Normalize       │
        │                                                   │
        └───────────────────────────────────────────────────┘
                                    │
                                    ↓
        ┌───────────────────────────────────────────────────┐
        │           DATALOADER & BATCHING LAYER             │
        ├───────────────────────────────────────────────────┤
        │                                                   │
        │  Train DataLoader (shuffle=True)                  │
        │    └─→ Batch size: 16                             │
        │    └─→ Mixup augmentation applied                 │
        │                                                   │
        │  Val DataLoader (shuffle=False)                   │
        │    └─→ Batch size: 16                             │
        │    └─→ Clean images (no augmentation)             │
        │                                                   │
        │  Stratified Holdout (if val < 24 samples)         │
        │    └─→ Split train → additional val (15%)         │
        │    └─→ Preserve class balance                     │
        │                                                   │
        └───────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
        ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
        │   ResNet50   │  │ EfficientNet │  │  ViT-B/16    │
        │   TRAINING   │  │   B0 TRAIN   │  │   TRAINING   │
        └──────────────┘  └──────────────┘  └──────────────┘
             (Model architecture diagram below)
```

---

## 2. MODEL ARCHITECTURE DETAILS

### ResNet50 Architecture

```
Input Image (B, 3, 224, 224)
         │
    ┌────────────┐
    │  Conv1(7×7, 64)
    │  BatchNorm
    │  ReLU
    │  MaxPool(3×3)
    └────────────┘
         │ (B, 64, 56, 56)
    ┌────────────────────┐
    │ Layer1 (3 blocks)  │  Residual blocks with skip connections
    │ 64 → 64 channels   │
    └────────────────────┘
         │ (B, 64, 56, 56)
    ┌────────────────────┐
    │ Layer2 (4 blocks)  │  Residual blocks with downsampling
    │ 64 → 128 channels  │
    └────────────────────┘
         │ (B, 128, 28, 28)
    ┌────────────────────┐
    │ Layer3 (6 blocks)  │  Residual blocks with downsampling
    │ 128 → 256 channels │
    └────────────────────┘
         │ (B, 256, 14, 14)
    ┌────────────────────┐
    │ Layer4 (3 blocks)  │  Residual blocks with downsampling
    │ 256 → 512 channels │
    └────────────────────┘
         │ (B, 512, 7, 7)
    ┌────────────┐
    │ AvgPool    │
    │ 7×7 → 1×1  │
    └────────────┘
         │ (B, 2048)
    ┌────────────────────────┐
    │ [REPLACE] fc layer     │
    │ Linear(2048, num_cls)  │  ← Only this trained in stage 1
    └────────────────────────┘
         │ (B, num_classes)
         ↓
    Logits output

Backbone Freeze (Stage 1):
  └─→ conv1, bn1, layer1, layer2, layer3, layer4 frozen
  └─→ Only fc trainable
```

### EfficientNet-B0 Architecture

```
Input Image (B, 3, 224, 224)
         │
    ┌──────────────────────┐
    │ Stem: Conv(3×3)      │
    │       BatchNorm       │
    │       SiLU (Swish)    │
    └──────────────────────┘
         │ (B, 32, 112, 112)
    ┌──────────────────────┐
    │ MBConv Blocks        │  Mobile Inverted Bottleneck
    │ Stages: 1-7          │  with SE blocks (squeeze-excitation)
    │ Expanding ratios     │  Depthwise separable convolutions
    │ Kernel sizes: 3, 5   │
    │ Strides vary         │
    └──────────────────────┘
         │ (progressive downsampling)
         │ (B, 1280, 7, 7)
    ┌──────────────────────┐
    │ Head: Conv(1×1)      │
    │       BatchNorm       │
    │       SiLU            │
    └──────────────────────┘
         │ (B, 1280, 7, 7)
    ┌────────────┐
    │ AvgPool    │
    │ 7×7 → 1×1  │
    └────────────┘
         │ (B, 1280)
    ┌──────────────────────────────┐
    │ [REPLACE] classifier[-1]     │
    │ Linear(1280, num_classes)    │  ← Only this trained in stage 1
    └──────────────────────────────┘
         │ (B, num_classes)
         ↓
    Logits output

Backbone Freeze (Stage 1):
  └─→ features (all MBConv layers) frozen
  └─→ Only classifier head trainable
```

### Vision Transformer (ViT-B/16) Architecture

```
Input Image (B, 3, 224, 224)
         │
    ┌────────────────────┐
    │ Patch Embedding    │
    │ 16×16 patches      │  224/16 = 14×14 = 196 patches
    │ Linear projection  │  768-dim embeddings
    │ (B, 196, 768)      │
    │                    │
    │ Prepend [CLS]      │  Special token for classification
    │ (B, 197, 768)      │
    │                    │
    │ Add position emb.  │  Learnable positional embeddings
    └────────────────────┘
         │ (B, 197, 768)
    ┌─────────────────────────────┐
    │ Transformer Encoder        │
    │ 12 layers of:              │
    │  - Multi-head attention    │  12 heads
    │  - LayerNorm               │
    │  - MLP (feedforward)       │
    │  - Residual connections    │
    └─────────────────────────────┘
         │ (B, 197, 768)
    ┌──────────────────────────┐
    │ Extract [CLS] token      │
    │ (B, 768)                 │
    └──────────────────────────┘
         │ (B, 768)
    ┌──────────────────────────────┐
    │ [REPLACE] heads.head         │
    │ Linear(768, num_classes)     │  ← Only this trained in stage 1
    └──────────────────────────────┘
         │ (B, num_classes)
         ↓
    Logits output

Backbone Freeze (Stage 1):
  └─→ encoder (transformer blocks) frozen
  └─→ Only heads.head trainable
```

---

## 3. TRAINING LOOP - EPOCH-BY-EPOCH FLOW

```
┌──────────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP (train.py)                      │
└──────────────────────────────────────────────────────────────────┘

FOR epoch = 1 to max_epochs:

┌─ STAGE 1: PREPARE EPOCH ─────────────────────────────────────────┐
│                                                                   │
│  1. Backbone frozen?                                             │
│     ├─ if epoch <= freeze_epochs:                                │
│     │  └─→ Set all backbone params: requires_grad = False        │
│     │      (Only classifier head trainable)                      │
│     └─ else:                                                      │
│        └─→ Unfreeze everything: requires_grad = True             │
│                                                                   │
│  2. Warmup Learning Rate                                         │
│     ├─ if epoch <= 3:                                            │
│     │  └─→ lr = base_lr × (0.4 + 0.6 × epoch/3)                 │
│     │      Gradual ramp to base_lr                               │
│     └─ else:                                                      │
│        └─→ Use current scheduler lr                              │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─ STAGE 2: TRAINING FORWARD PASS ────────────────────────────────┐
│                                                                   │
│  model.train()                                                    │
│                                                                   │
│  FOR each batch in train_loader:                                 │
│                                                                   │
│     images, labels = batch  # (B, 3, 224, 224), (B,)            │
│                                                                   │
│     A. Mixup Data Augmentation                                   │
│        ├─ λ ~ Beta(α, α)    [α depends on model]               │
│        ├─ Select random permutation of batch                     │
│        ├─ mixed_x = λ × x + (1-λ) × x_shuffled                  │
│        └─ Return: mixed_images, labels_a, labels_b, λ           │
│                                                                   │
│     B. Forward Pass (Automatic Mixed Precision)                  │
│        ├─ with torch.amp.autocast("cuda"):                       │
│        │  └─→ logits = model(mixed_images)  # (B, num_classes)  │
│        │                                                         │
│        └─ Mixup Loss:                                            │
│           loss = λ × CE_loss(logits, labels_a) +                │
│                  (1-λ) × CE_loss(logits, labels_b)              │
│                                                                   │
│     C. Backpropagation                                           │
│        ├─ optimizer.zero_grad()                                  │
│        ├─ scaler.scale(loss).backward()                          │
│        │  (Gradient scaling for mix-precision training)          │
│        │                                                         │
│        ├─ Gradient Clipping (optional):                          │
│        │  ├─ if grad_clip_norm > 0:                             │
│        │  └─→ torch.nn.utils.clip_grad_norm_(params, clip_norm) │
│        │                                                         │
│        └─ Optimizer Step:                                        │
│           ├─ scaler.step(optimizer)                              │
│           └─ scaler.update()                                     │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─ STAGE 3: EVALUATION ───────────────────────────────────────────┐
│                                                                   │
│  model.eval()  # Disable dropout, use batch norm statistics      │
│                                                                   │
│  train_acc, train_loss = evaluate(train_metric_loader)           │
│    └─→ Forward pass on CLEAN train samples (no mixup)            │
│    └─→ Compute accuracy and cross-entropy loss                   │
│                                                                   │
│  val_acc, val_loss = evaluate(val_loader)                        │
│    └─→ Forward pass on validation set                            │
│    └─→ Compute accuracy and loss                                 │
│                                                                   │
│  Metrics recorded:                                                │
│    history["epoch"].append(epoch)                                 │
│    history["train_loss"].append(train_loss)                       │
│    history["val_loss"].append(val_loss)                           │
│    history["train_acc"].append(train_acc)                         │
│    history["val_acc"].append(val_acc)                             │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─ STAGE 4: LEARNING RATE SCHEDULING ─────────────────────────────┐
│                                                                   │
│  if epoch > warmup_epochs (epoch > 3):                            │
│     └─→ scheduler.step(val_loss)                                 │
│                                                                   │
│         ReduceLROnPlateau Logic:                                  │
│         ├─ If val_loss improved: reset patience counter          │
│         ├─ Else: increment patience counter                      │
│         ├─ If patience > 2: multiply lr by factor=0.7            │
│         └─ Min lr: base_lr × 0.1                                  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─ STAGE 5: EARLY STOPPING LOGIC ─────────────────────────────────┐
│                                                                   │
│  A. Train/Validation Gap Guard                                   │
│     gap = train_acc - val_acc                                    │
│     ├─ if gap > max_train_val_gap AND epoch >= gap_guard_start: │
│     │  ├─ epochs_with_large_gap += 1                            │
│     │  ├─ if epochs_with_large_gap >= max_large_gap_epochs:     │
│     │  │  └─→ STOP TRAINING (avoid divergence/overfitting)      │
│     │  └─ Rationale: Prevent encoder from memorizing noise      │
│     └─ else:                                                      │
│        └─→ epochs_with_large_gap = 0 (reset)                    │
│                                                                   │
│  B. Patience-based Early Stopping                                │
│     ├─ if val_acc > best_acc:                                    │
│     │  ├─→ Save checkpoint (model_state, class_to_idx)           │
│     │  ├─→ best_acc = val_acc                                    │
│     │  └─→ epochs_without_improve = 0 (reset)                   │
│     └─ else:                                                      │
│        ├─→ epochs_without_improve += 1                           │
│        ├─→ if epochs_without_improve >= early_stop_patience:    │
│        │  └─→ STOP TRAINING (no improvement)                     │
│        └─→ Print "wait X/{patience}"                             │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─ STAGE 6: LOGGING & DISPLAY ────────────────────────────────────┐
│                                                                   │
│  Print epoch summary:                                             │
│  $ Epoch  5 | TrLoss  0.3421 | VaLoss  0.4125 |                  │
│           | TrAcc   0.7890 | VaAcc   0.7650 | [SAVED]            │
│                                                                   │
│  Diagnostic output every N epochs (epochs 4-7):                   │
│  [DIAG] lr_before=0.0002 lr_after=0.00014                        │
│         grad_norm_avg=0.123 grad_norm_max=0.456                  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

END FOR

┌─ POST-TRAINING ─────────────────────────────────────────────────┐
│                                                                   │
│  Save final artifacts:                                            │
│  ├─ checkpoint: models/{model}_epillid_best.pt                   │
│  ├─ metrics: models/{model}_epillid_best.metrics.json             │
│  ├─ history: models/{model}_epillid_history.json                  │
│  └─ curves: models/{model}_training_curves.png                    │
│                                                                   │
│  Plot training curves (with EMA smoothing):                        │
│  ├─ X-axis: epoch                                                 │
│  ├─ Y-axis (left): Loss (train vs val)                           │
│  ├─ Y-axis (right): Accuracy (train vs val)                      │
│  ├─ Highlight best epoch (minimum val_loss)                      │
│  └─ Show train/val gap as shaded region                          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. INFERENCE PIPELINE

```
┌──────────────────────────────────────────────────────────────────┐
│              INFERENCE (inference.py, gui_tk.py)                 │
└──────────────────────────────────────────────────────────────────┘

Step 1: Load Model from Checkpoint
┌────────────────────────────────────────────┐
│ checkpoint_path = "models/resnet50_best.pt"│
│                                            │
│ _get_or_load_model(                        │
│   model_name="resnet50",                   │
│   checkpoint_path=...,                     │
│   device="cuda"                            │
│ )                                          │
│                                            │
│ Check cache: (model_name, path, device)?  │
│ ├─ YES: Return cached model                │
│ └─ NO: Load checkpoint                     │
│     ├─ model = create_model(...)           │
│     ├─ model.load_state_dict(...)          │
│     ├─ model.to(device)                    │
│     ├─ model.eval()                        │
│     ├─ Cache model                         │
│     └─ Return model                        │
│                                            │
│ Also load: class_to_idx from checkpoint    │
│            {class_name: idx}               │
└────────────────────────────────────────────┘
         │
         ↓
Step 2: Preprocess Image
┌────────────────────────────────────────────┐
│ image_path = "path/to/query.jpg"           │
│                                            │
│ 1. pil_loader(path)                        │
│    └─→ PIL Image (RGB)                     │
│                                            │
│ 2. focus_on_object(img, scale=0.85)        │
│    ├─→ Compute crop region (center)        │
│    ├─→ Center crop to remove background    │
│    └─→ PIL Image (cropped)                 │
│                                            │
│ 3. build_transforms(train=False)(img)      │
│    ├─→ Resize(224×224)                     │
│    ├─→ ToTensor: PIL → torch.Tensor        │
│    │   Values: [0, 1] → (B, 3, 224, 224)   │
│    └─→ Normalize (ImageNet mean/std)       │
│        Values: (-2.1 to +2.6 range)        │
│                                            │
│ Return: tensor (1, 3, 224, 224)            │
└────────────────────────────────────────────┘
         │
         ↓
Step 3: Model Forward Pass
┌────────────────────────────────────────────┐
│ with torch.no_grad():                      │
│   logits = model(tensor)                   │
│   # (1, num_classes)                       │
│                                            │
│ Example: num_classes = 8                   │
│ logits = [0.123, -0.45, 0.891, -0.12, ..] │
└────────────────────────────────────────────┘
         │
         ↓
Step 4: Generate Prediction
┌────────────────────────────────────────────┐
│ probabilities = softmax(logits)            │
│ # Sum to 1.0, all positive                 │
│                                            │
│ pred_idx = argmax(logits)                  │
│ confidence = max(probabilities)            │
│                                            │
│ pred_class = inv_class_to_idx[pred_idx]    │
│ # e.g., "cefadroxil_500mg_0.5g"            │
│                                            │
│ Return ComparisonResult(                   │
│   predicted_class="cefadroxil_500mg_0.5g", │
│   confidence=0.85                          │
│ )                                          │
└────────────────────────────────────────────┘
```

---

## 5. ENSEMBLE COMPARISON PROCESS

```
Query Image (User Upload)
         │
         ├─────────────────────────────────────────┐
         │                                         │
         ↓                                         ↓
    ┌─────────────┐                          ┌─────────────┐
    │   ResNet50  │                          │ Efficientnet│
    │  Inference  │                          │  Inference  │
    └─────────────┘                          └─────────────┘
         │                                         │
         ├─ pred_class: "class_A"                  ├─ pred_class: "class_B"
         ├─ confidence: 0.75                       ├─ confidence: 0.68
         └─ probs: [0.75, 0.15, 0.10, ...]        └─ probs: [0.68, 0.20, 0.12, ...]
                                                   │
         ↓                                         ↓
    ┌─────────────────────────────────────────────────────┐
    │           ENSEMBLE VOTING                           │
    ├─────────────────────────────────────────────────────┤
    │                                                     │
    │  Average probabilities across models:              │
    │  ensemble_probs = mean([                           │
    │    model1_probs,                                    │
    │    model2_probs,                                    │
    │    model3_probs                                     │
    │  ])                                                 │
    │                                                     │
    │  ensemble_pred = argmax(ensemble_probs)            │
    │                                                     │
    │  Return: (ensemble_prediction, individual_results) │
    │                                                     │
    └─────────────────────────────────────────────────────┘
```

---

## 6. DATA DIRECTORY STRUCTURE - DISCOVERY LOGIC

```
Current Working Directory
│
├─── data_aligned/              ← PREFERRED (richer)
│    ├── train/
│    │   ├── class_0/
│    │   │   ├── img_001.jpg
│    │   │   ├── img_002.jpg
│    │   │   └── ...
│    │   ├── class_1/
│    │   │   └── ...
│    │   └── class_N/
│    ├── val/
│    │   ├── class_0/
│    │   ├── class_1/
│    │   └── ...
│    └── test/
│        ├── cefadroxil_500mg_0.5g/
│        ├── golddicron_30mg/
│        └── ...
│
└─── data/                      ← FALLBACK
     ├── train/
     ├── val/
     └── test/

Logic (pipeline.py::discover_data_dir):
  1. Check --data-dir argument (if provided)
  2. Else: Check candidates [data_aligned, data]
  3. Validate: Each has train/, val/, test/ subfolders
  4. Prefer data_aligned (more classes)
  5. If both valid: pick the one with more classes
  6. If neither valid: raise FileNotFoundError
```

---

## 7. CLASS MAPPING PRESERVATION DURING TRAINING & INFERENCE

```
TRAINING FLOW:
┌──────────────────────────────────────────┐
│ Load train dataset                       │
│ train_ds = PillImageDataset("data/train")│
│ class_to_idx = {                         │
│   "cefadroxil_500mg_0.5g": 0,           │
│   "golddicron_30mg": 1,                  │
│   "kavasdin_5_5mg": 2,                   │
│   ...                                    │
│ }                                        │
└──────────────────────────────────────────┘
                  │
                  ↓
┌──────────────────────────────────────────┐
│ TRAIN MODEL                              │
└──────────────────────────────────────────┘
                  │
                  ↓
┌──────────────────────────────────────────┐
│ SAVE CHECKPOINT                          │
│ torch.save({                             │
│   "model_state_dict": model.state_dict(),│
│   "num_classes": 8,                      │
│   "class_to_idx": {                      │  ← CRITICAL!
│     "cefadroxil_500mg_0.5g": 0,         │
│     "golddicron_30mg": 1,                │
│     ...                                  │
│   },                                     │
│   "val_acc": 0.75                        │
│ }, checkpoint_path)                      │
└──────────────────────────────────────────┘


INFERENCE FLOW:
┌──────────────────────────────────────────┐
│ LOAD CHECKPOINT                          │
│ state = torch.load("...best.pt")         │
│ ckpt_class_to_idx = state["class_to_idx"]│
│ num_classes = state["num_classes"]       │
└──────────────────────────────────────────┘
                  │
                  ↓
┌──────────────────────────────────────────┐
│ CREATE MODEL with num_classes            │
│ model = create_model(                    │
│   "resnet50",                            │
│   num_classes=8                          │
│ )                                        │
│ model.load_state_dict(...)               │
└──────────────────────────────────────────┘
                  │
                  ↓
┌──────────────────────────────────────────┐
│ FORWARD PASS                             │
│ logits = model(image)  # (1, 8)         │
│ pred_idx = argmax(logits)  # int 0-7     │
│                                          │
│ inv_mapping = {v: k for k, v in         │
│   ckpt_class_to_idx.items()}             │
│ pred_class = inv_mapping[pred_idx]       │
│ # "cefadroxil_500mg_0.5g"                │
└──────────────────────────────────────────┘

MISMATCH SCENARIO:
  Training: 8 classes (original structure)
  Inference: 7 classes (after user deletes one)
  
  Problem: checkpoint.class_to_idx has 8 classes,
           but model trained for 8 output neurons
           
  Solution (in code):
    ├─ Load ckpt_class_to_idx from checkpoint
    ├─ If mismatch detected: warn user
    ├─ Use ckpt_class_to_idx for mapping (safe)
    └─ Recommend: retrain if structure changed
```

---

## 8. CHECKPOINT FORMAT (Visual)

```
models/resnet50_epillid_best.pt
│
└─→ Python dict (after torch.load)
   │
   ├─ "model_state_dict"
   │  └─→ Dictionary of all model parameter tensors
   │     ├─ "conv1.weight": shape (64, 3, 7, 7)
   │     ├─ "conv1.bias": shape (64,)
   │     ├─ "layer1.0.conv1.weight": shape (64, 64, 1, 1)
   │     ├─ "layer1.0.conv1.bias": shape (64,)
   │     ├─ ... (hundreds of layer parameters) ...
   │     ├─ "layer4.2.conv3.weight": shape (512, 512, 3, 3)
   │     └─ "fc.weight": shape (num_classes, 2048)
   │        "fc.bias": shape (num_classes,)
   │
   ├─ "num_classes"
   │  └─→ Integer, e.g., 8
   │
   ├─ "class_to_idx"
   │  └─→ Dictionary mapping class names to indices
   │     ├─ "cefadroxil_500mg_0.5g": 0
   │     ├─ "golddicron_30mg": 1
   │     ├─ "kavasdin_5_5mg": 2
   │     ├─ "panactol_500mg": 3
   │     ├─ "sergurop_10mg": 4
   │     ├─ "thuoc_chua_dinh_danh_0": 5
   │     ├─ "thuoc_chua_dinh_danh_1": 6
   │     └─ "thuoc_ngoai_don_class_107": 7
   │
   └─ "val_acc"
      └─→ Float, e.g., 0.75
```

---

**Legend:**
- `─→` Flow direction
- `├─` Tree branch
- `└─` Tree end
- `│` Continuation line
- `┌┐└┘` Box corners
- `─` Horizontal line
- `▶` Highlighted point

