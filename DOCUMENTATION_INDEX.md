# THUOC PROJECT EXPLORATION - COMPLETE DOCUMENTATION INDEX

**Date**: March 24, 2026  
**Project**: THUOC - Pill Classification from Images  
**Analysis Type**: Full codebase exploration with architecture diagrams

---

## 📚 DOCUMENTATION FILES CREATED

All files have been created in the root of your THUOC project folder:

### 1. **PROJECT_ANALYSIS.md** (Comprehensive)
   - **Length**: ~1,000+ lines
   - **Contains**:
     - Complete file mapping (root + src/)
     - Data pipeline steps (discovery → loading → augmentation → batching)
     - Training algorithm overview with timing diagrams
     - Model architecture details (ResNet50, EfficientNet, ViT)
     - Inference process (single model + ensemble)
     - Requirements list with version explanations
     - Output artifacts breakdown
     - Summary tables
   - **Use when**: You need complete technical deep-dive
   - **Key Sections**: Sections 1-8 cover different aspects

### 2. **FILE_MAPPING_DICTIONARY.md** (Quick Reference)
   - **Length**: ~500 lines
   - **Format**: Structured JSON-style dictionary
   - **Contains**:
     - Root files (run_all.py, train_cli.py, etc.)
     - src/ folder modules (detailed dataclass fields + functions)
     - Data pipeline flow
     - Training flow
     - Inference flow
     - Key concepts glossary
     - Command reference
   - **Use when**: You want quick lookup of what each file does
   - **Key Features**: 
     - JSON reference format
     - Function signatures
     - Return types documented
     - Important constants listed

### 3. **ARCHITECTURE_DIAGRAMS.md** (Visual Guide)
   - **Length**: ~800 lines
   - **Contains**:
     - System architecture overview (ASCII boxes)
     - Model architecture details (ResNet50, EfficientNet-B0, ViT-B/16)
     - Training loop epoch-by-epoch flow
     - Inference pipeline step-by-step
     - Ensemble comparison process
     - Data directory discovery logic
     - Class mapping preservation strategy
     - Checkpoint format visualization
   - **Use when**: You want visual understanding of flows
   - **Format**: ASCII art diagrams with annotations

---

## 🎯 HOW TO USE THIS DOCUMENTATION

### Scenario 1: "I want to understand the entire project quickly"
**→ Read in this order:**
1. Start with **ARCHITECTURE_DIAGRAMS.md** section 1 (System Overview)
2. Then **PROJECT_ANALYSIS.md** sections 1-2 (File mapping + Data pipeline)
3. Finally skim **FILE_MAPPING_DICTIONARY.md** for quick lookup

**Time: ~15-20 minutes**

### Scenario 2: "I need to understand how training works"
**→ Read:**
1. **ARCHITECTURE_DIAGRAMS.md** section 3 (Training Loop)
2. **PROJECT_ANALYSIS.md** section 3 (Training Algorithm)
3. Check **FILE_MAPPING_DICTIONARY.md** for train.py details

**Time: ~10 minutes**

### Scenario 3: "I need to modify or add a model"
**→ Read:**
1. **PROJECT_ANALYSIS.md** section 4 (Model Architectures)
2. **ARCHITECTURE_DIAGRAMS.md** section 2 (Model details)
3. **FILE_MAPPING_DICTIONARY.md** → models.py section

**Time: ~5-10 minutes**

### Scenario 4: "I need to make inference work"
**→ Read:**
1. **ARCHITECTURE_DIAGRAMS.md** section 4 (Inference Pipeline)
2. **PROJECT_ANALYSIS.md** section 5 (Inference Flow)
3. **FILE_MAPPING_DICTIONARY.md** → inference.py section

**Time: ~5-10 minutes**

### Scenario 5: "I'm debugging and need to find what's in a checkpoint"
**→ Read:**
1. **ARCHITECTURE_DIAGRAMS.md** section 8 (Checkpoint Format)
2. **PROJECT_ANALYSIS.md** section 7 (Output Artifacts)

**Time: ~2-3 minutes**

---

## 🔑 KEY INSIGHTS FROM EXPLORATION

### 1. **Data Handling**
- Dataset structure: `train/val/test/` each with class folders
- If val set < 24 samples: stratified holdout (15%) from train is auto-created
- All images: 224×224, RGB, normalized with ImageNet stats
- Training aug: Mixup + ColorJitter + Flip + Rotation
- Eval: No augmentation, just resize + normalize

### 2. **Training Approach**
- **Stage 1**: Freeze backbone (1-5 epochs), train only classifier head
- **Stage 2**: Unfreeze all layers, continue training
- **Warmup**: Linear ramp from 40% to 100% of base_lr in first 3 epochs
- **Loss**: CrossEntropyLoss with class-level label smoothing
- **Optimizer**: AdamW with weight decay (L2 regularization)
- **Scheduler**: ReduceLROnPlateau (multiply lr by 0.7 if plateau for 2 epochs)
- **Early stop**: Either patience=8 epochs no improvement OR train-val gap too large

### 3. **Three Models Compared**
```
ResNet50:         Larger (>90M params), stable, good baseline
EfficientNet-B0:  Lighter (<7M params), efficient, fast
ViT-B/16:         Attention-based, different inductive bias, modern
```

### 4. **Class Mapping is Critical**
- **Problem**: During inference, class indices must match checkpoint classes
- **Solution**: Save `class_to_idx` dict in every checkpoint
- **Safety**: If mismatch detected, use checkpoint mapping but warn user
- **Recommendation**: Retrain if data structure changes significantly

### 5. **Hyperparameter Tuning Strategy**
- Pre-tuned `OPTIMAL_CONFIGS` in `optimal_configs.py` for each model
- ViT needs lower LR (5e-5 vs 2e-4), higher regularization, more backbone freezing
- Small datasets (<64 samples): auto-adjust regularization downward
- Review tool (`review_terminal.py`) suggests changes during training

### 6. **Output Artifacts**
```
models/
├─ {model}_epillid_best.pt          (Checkpoint)
├─ {model}_epillid_best.metrics.json (Best metrics)
├─ {model}_epillid_history.json      (Per-epoch metrics)
├─ {model}_training_curves.png       (Loss/Acc plots)
├─ evaluation_summary.csv            (Comparison table)
└─ evaluation_comparison.png         (Bar chart)
```

### 7. **GUI & Inference Features**
- Compare 2 pill images (sample vs. query)
- Display: similarity score, color score, size score, texture score
- Show: medicine metadata from CSV (if available)
- Ensemble voting (optional): average probabilities across 3 models

---

## 📋 QUICK COMMAND REFERENCE

```bash
# Complete pipeline (train all 3 models + evaluate)
python run_all.py

# Train single model
python run_all.py --model resnet50 --epochs 20

# Only evaluate (no training)
python run_all.py --compare-only

# Use CPU instead of GPU
python run_all.py --device cpu

# Launch GUI
python run_gui.py

# CLI training modes
python train_cli.py --mode all              # Full pipeline
python train_cli.py --mode single --epochs 30  # Single model
python train_cli.py --mode optimize --rounds 5  # Hyperparameter tuning
```

---

## 🧩 PROJECT STRUCTURE AT-A-GLANCE

```
THUOC/
├── README.md                      ← User guide (Vietnamese)
├── requirements.txt               ← 8 dependencies
├── optimal_configs.py             ← Pre-tuned hyperparams
├── run_all.py                     ← Main entry point
├── train_cli.py                   ← CLI wrapper
├── run_gui.py                     ← GUI launcher
├── review_terminal.py             ← Interactive tuning
│
├── src/
│   ├── train.py                   ← Training loop (700 lines)
│   ├── models.py                  ← 3 model builders
│   ├── features.py                ← Dataset + augmentation
│   ├── pipeline.py                ← Orchestration
│   ├── inference.py               ← Prediction engine
│   ├── evaluate_report.py         ← Metrics & CSV export
│   ├── gui_tk.py                  ← Tkinter GUI
│   ├── build_epillid_data.py      ← Data integration
│   ├── metadata.py                ← CSV parsing
│   └── self_learning.py           ← Feedback logging
│
├── data/ or data_aligned/         ← Training data
│   ├── train/{class_0, class_1, ...}
│   ├── val/{class_0, class_1, ...}
│   └── test/{8 pill classes}
│
├── models/                        ← Checkpoints & results
│   ├── resnet50_epillid_best.pt
│   ├── efficientnet_b0_epillid_best.pt
│   ├── vit_b_16_epillid_best.pt
│   ├── {model}_history.json
│   ├── {model}_training_curves.png
│   ├── evaluation_summary.csv
│   └── reports/
│
└── demo_images/                   ← Sample images for GUI
    └── {8 pill classes}

[+ NEW DOCUMENTATION]
├── PROJECT_ANALYSIS.md            ✨ Created!
├── FILE_MAPPING_DICTIONARY.md     ✨ Created!
└── ARCHITECTURE_DIAGRAMS.md       ✨ Created!
```

---

## 🔗 CROSS-REFERENCES

### If you want to understand **Data Flow**:
- Read: PROJECT_ANALYSIS.md section 2
- Visualize: ARCHITECTURE_DIAGRAMS.md section 6
- Code: src/features.py + src/train.py::create_dataloaders()

### If you want to understand **Model Architecture**:
- Read: PROJECT_ANALYSIS.md section 4
- Visualize: ARCHITECTURE_DIAGRAMS.md section 2
- Code: src/models.py

### If you want to understand **Training Loop**:
- Read: PROJECT_ANALYSIS.md section 3
- Visualize: ARCHITECTURE_DIAGRAMS.md section 3
- Code: src/train.py::train() function

### If you want to understand **Inference**:
- Read: PROJECT_ANALYSIS.md section 5
- Visualize: ARCHITECTURE_DIAGRAMS.md section 4
- Code: src/inference.py

### If you want to understand **Requirements**:
- Read: PROJECT_ANALYSIS.md section 6
- Check: requirements.txt

### If you want to understand **Output**:
- Read: PROJECT_ANALYSIS.md section 7
- Code: src/pipeline.py + src/evaluate_report.py

---

## 💡 NOTES FOR NEXT STEPS

1. **For Training Improvement**:
   - Review OPTIMAL_CONFIGS in optimal_configs.py
   - Use review_terminal.py to monitor training in real-time
   - Increase dataset size (small test set ~10 samples is limiting)

2. **For Inference**:
   - Models are cached per session (no reload overhead in GUI)
   - Class mapping is validated at load time
   - Ensemble voting available but not default

3. **For Data**:
   - Stratified holdout auto-created if val too small
   - Support for both copy/hardlink/symlink file operations
   - Metadata CSV optional but enriches comparison results

4. **For Deployment**:
   - Checkpoints include class_to_idx for portability
   - GPU auto-detected, fallback to CPU if needed
   - Mixed precision training for faster GPU compute

---

## ✅ DOCUMENTATION COMPLETENESS CHECKLIST

- [x] All 13 Python files mapped with functions
- [x] Data pipeline steps explained (discovery → loading → augmentation → batching)
- [x] Training algorithm with loss, optimizer, scheduler details
- [x] Three model architectures documented (ResNet50, EfficientNet, ViT)
- [x] Inference process with caching explained
- [x] Requirements list (8 packages with versions)
- [x] Output artifacts breakdown
- [x] Visual ASCII diagrams for all major flows
- [x] File-by-file function reference
- [x] Command cheat sheet included
- [x] Cross-references between documents
- [x] Key insights summarized

---

**Last Updated**: 2026-03-24  
**Analysis Scope**: Full codebase exploration  
**Documentation Files**: 3 comprehensive guides  
**Total Lines**: 2,000+ lines of detailed documentation
