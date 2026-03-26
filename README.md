# THUOC - Project Deep Dive (Human + AI Agent Friendly)

THUOC la he thong phan loai vien thuoc tu anh, duoc thiet ke theo pipeline ro rang de:
- huan luyen nhieu backbone,
- danh gia theo cung bo metric,
- tao bao cao de so sanh,
- va mo rong de review/tuning theo vong lap.

Tai lieu nay viet cho 2 doi tuong:
- Nha phat trien con nguoi: can hieu nhanh he thong de chay, debug, dong gop.
- AI agent: can hieu ro contract dau vao/dau ra, luong xu ly, va diem can sua trong code.

## 1. Executive Summary

Neu ban chi co 2 phut, hay nho 5 diem sau:

1. Entrypoint full pipeline la `run_all.py`.
2. Core business logic nam trong `src/` (data, models, training, orchestration, evaluation, inference).
3. He thong train 3 model: `resnet50`, `efficientnet_b0`, `vit_b_16`.
4. Output chinh nam trong `models/` (checkpoint, metrics, history, report, charts).
5. Bo unit test nhanh nam trong `tests/`.

## 2. How To Run

### 2.1 Cai dat

```bash
pip install -r requirements.txt
```

### 2.2 Full pipeline (khuyen dung)

```bash
python run_all.py
```

### 2.3 Cac lenh thuong dung

```bash
# Train 1 model
python train_cli.py --mode single --model resnet50 --epochs 28

# Chay optimize nhieu vong (train -> review -> de xuat config)
python train_cli.py --mode optimize --rounds 3 --epochs 12

# Chi evaluate checkpoint da co
python run_all.py --compare-only

# Chay bang CPU
python run_all.py --device cpu
```

### 2.4 Test nhanh

```bash
python -m pytest tests/test_features.py tests/test_inference_utils.py tests/test_metadata.py -q
```

## 3. System Architecture

```mermaid
flowchart LR
  A[Dataset train/val/test] --> B[Data module]
  B --> C[Training module]
  C --> D[Best checkpoint per model]
  D --> E[Evaluation module]
  E --> F[Weighted ensemble]
  F --> G[Reports CSV/JSON/PNG]
```

### 3.1 Module boundaries

1. Data layer:
  Chuan hoa du lieu, transform anh, doc metadata.
2. Model layer:
  Tao backbone, load checkpoint, dong bo class mapping.
3. Training layer:
  Train loop, validation, early-stop, mixup, EMA, TTA eval.
4. Orchestration layer:
  Dieu phoi toan bo quy trinh train/evaluate/report.
5. Evaluation layer:
  Tinh accuracy/macro-F1, tao bang tong hop va bieu do.
6. Inference layer:
  So sanh 2 anh thuoc, ket hop deep feature + heuristic scores.
7. Learning feedback layer:
  Ghi feedback sai/dung cho cac vong cai tien sau.

## 4. Folder-by-Folder Explanation

## 4.1 Root level

| Path | Vai tro | Dau vao | Dau ra |
|---|---|---|---|
| `run_all.py` | Script chay full pipeline | data dir + config | train + evaluate + report |
| `train_cli.py` | CLI da che do (`all`, `single`, `optimize`) | args tu command line | goi dung module ben duoi |
| `THUOC_Colab_Train_Evaluate.ipynb` | Notebook chay tren Colab | Drive data + runtime GPU | artifacts cho demo/report |
| `requirements.txt` | Dependency list | - | moi truong chay thong nhat |
| `AGENTS.md` | Huong dan tac nghiep cho AI coding agent | - | quy tac phat trien nhat quan |
| `README.md` | Tai lieu tong quan va deep dive | - | onboarding cho nguoi/AI agent |

## 4.2 `src/` (Core code)

### `src/data/`

| File | Chuc nang | Mo ta ngan |
|---|---|---|
| `features.py` | Dataset + transforms | Dinh nghia `PillImageDataset`, augmentation train, transform eval, image statistics |
| `metadata.py` | Metadata matching | Parse CSV thuoc, tokenize text, map class name -> metadata record |
| `build_epillid_data.py` | Build aligned dataset | Chuyen raw split thanh cau truc thu muc train/val/test theo class |
| `__init__.py` | Package marker | Cho phep import module tu `src.data` |

### `src/models/`

| File | Chuc nang | Mo ta ngan |
|---|---|---|
| `resnet50.py` | Backbone builder | Tao model ResNet50 phu hop so lop |
| `efficientnet_b0.py` | Backbone builder | Tao model EfficientNet-B0 |
| `vit_b_16.py` | Backbone builder | Tao model ViT-B/16 |
| `model_factory.py` | Factory + checkpoint utils | Tao model theo ten, fallback offline, load checkpoint/class mapping |
| `__init__.py` | Package marker | Import convenience |

### `src/training/`

| File | Chuc nang | Mo ta ngan |
|---|---|---|
| `train.py` | Training engine | Parse args, tao dataloader, train loop, val loop, save best, curves/history |
| `__init__.py` | Package marker | - |

### `src/evaluation/`

| File | Chuc nang | Mo ta ngan |
|---|---|---|
| `evaluate_report.py` | Evaluate + report | Chay model tren test, tinh metrics, ghi CSV, ve comparison chart |
| `__init__.py` | Package marker | - |

### `src/inference/`

| File | Chuc nang | Mo ta ngan |
|---|---|---|
| `inference.py` | Inference + image comparison | Predict class va tinh do tuong dong 2 anh (feature + color/size/texture) |
| `__init__.py` | Package marker | - |

### `src/orchestration/`

| File | Chuc nang | Mo ta ngan |
|---|---|---|
| `pipeline.py` | End-to-end orchestrator | Discover data, train nhieu model, evaluate tung model, ensemble, xuat summary |
| `__init__.py` | Package marker | - |

### `src/learning/`

| File | Chuc nang | Mo ta ngan |
|---|---|---|
| `self_learning.py` | Feedback logging | Luu phan hoi dung/sai, tao danh sach hard examples de retrain |
| `__init__.py` | Package marker | - |

## 4.3 `Review/` (optimization support)

| File | Chuc nang | Mo ta ngan |
|---|---|---|
| `review_terminal.py` | Vong lap review terminal | Huan luyen theo round, danh gia, de xuat tham so tiep theo |
| `optimal_configs.py` | Cac config toi uu | Bang hyperparameter cho tung backbone |
| `__init__.py` | Package marker | Dam bao import theo package (`Review.*`) |

## 4.4 Data and artifacts folders

| Folder | Vai tro |
|---|---|
| `data/` | Dataset goc/thu nghiem, co the khong dong nhat class giua split |
| `data_aligned/` | Dataset da align class mapping, uu tien dung de train |
| `demo_images/` | Anh mau de demo infer/compare |
| `models/` | Toan bo artifact sau train/evaluate |
| `tests/` | Unit tests cho data utils va inference helpers |

## 5. Algorithmic Deep Dive

## 5.1 Data pipeline

1. Nguon du lieu:
  `data_aligned/` duoc uu tien vi class names dong nhat giua train/val/test.
2. Train transform:
  random resized crop + flip + color jitter + normalize.
3. Eval transform:
  deterministic resize + center crop + normalize.
4. Dataset output tuple:
  `(image_tensor, class_idx, image_path)`.

## 5.2 Training algorithm

Training trong `src/training/train.py` co cac ky thuat sau:

1. Transfer learning:
  Khoi tao backbone pretrained (neu co internet/weights cache).
2. Label smoothing:
  Giam over-confident khi train classification.
3. Mixup:
  Tron 2 mau trong batch de tang generalization.
4. Gradient clipping:
  Han che gradient explosion.
5. Early stopping:
  Dung train khi metric khong cai thien hoac train-val gap qua lon.
6. EMA (Exponential Moving Average):
  Lam muot tham so de val metric on dinh hon.
7. TTA validation:
  Kiem tra voi nhieu bien the anh (flip/rotate) roi lay trung binh logits.

Cong thuc mixup co ban:

$$
x' = \lambda x_i + (1-\lambda)x_j, \quad
y' = \lambda y_i + (1-\lambda)y_j
$$

voi $\lambda \sim \text{Beta}(\alpha, \alpha)$.

## 5.3 Evaluation and ensemble

1. Moi model duoc evaluate doc lap tren test set.
2. Metric chinh: Accuracy va Macro-F1.
3. Ensemble weighted soft-voting:
  moi model dong gop xac suat co trong so theo chat luong validation.

Cong thuc tong quat:

$$
P_{ens}(c) = \frac{\sum_m w_m P_m(c)}{\sum_m w_m}
$$

Trong do $w_m$ la trong so cua model $m$.

## 5.4 Inference image-to-image comparison

Trong `src/inference/inference.py`, he thong ket hop 2 nhom tin hieu:

1. Deep feature similarity:
  Trich feature vector bang forward hook tai layer truoc classifier.
2. Hand-crafted signals:
  color score, size score, texture score.

Ket qua cuoi cung dung rule-based gating:
- class match,
- so feature dat nguong,
- va semantic penalty neu active ingredient group khong khop.

## 6. End-to-End Runtime Flow

Khi chay `python run_all.py`, luong xu ly nhu sau:

1. Resolve data dir (`data_aligned` uu tien).
2. Kiem tra split consistency train/val/test.
3. Train tung model theo `OPTIMAL_CONFIGS`.
4. Luu best checkpoint + metrics/history moi model.
5. Evaluate tung checkpoint tren test set.
6. Chay ensemble weighted.
7. Xuat report `csv/json/png` vao `models/` va `models/reports/latest/`.

## 7. Contracts For AI Agents

Section nay de AI agent doc va thao tac an toan hon.

1. Stable entrypoints:
  `run_all.py`, `train_cli.py`, `src/orchestration/pipeline.py`.
2. Dataset contract:
  split folders `train`, `val`, `test` voi cau truc `split/class/*.jpg`.
3. Checkpoint naming contract:
  `<model_name>_epillid_best.pt`.
4. Metrics files:
  `<model_name>_epillid_best.metrics.json` va `<model_name>_epillid_history.json`.
5. Report outputs:
  `models/evaluation_summary.csv` va `models/reports/latest/*`.
6. Safety notes:
  giu class mapping on dinh khi infer,
  khong gia dinh checkpoint va dataset co cung index space,
  uu tien su dung helper `load_checkpoint_class_to_idx`.

## 8. Common Failure Modes

1. Data split mismatch:
  train/val/test khac class folders -> train hoac evaluate sai mapping.
2. CUDA unavailable:
  project tu fallback CPU, nhung toc do se cham.
3. Offline pretrained download error:
  factory co fallback random init (tuy config).
4. Notebook local warning (`google.colab`):
  chi lien quan notebook runtime, khong phai loi source trong `src/`.
5. Large file push error:
  checkpoint `.pt` co the vuot gioi han GitHub.

## 9. Suggested Git Publishing Layout

De nguoi doc de tiep can khi len Git:

1. Giu README nay o root.
2. Dat 1 release note ngan trong Description cua repo.
3. Neu khong push checkpoint, ghi ro cach tai artifact hoac cach retrain.
4. Them hinh `evaluation_comparison.png` vao README neu muon showcase nhanh.

## 10. Final Checklist

1. Test nhanh da pass.
2. README da cap nhat.
3. Khong con loi syntax trong source Python.
4. Co day du report quan trong trong `models/`.
5. Kiem tra `git status` truoc khi push.

## 11. License

Du an nay su dung giay phep phi thuong mai cho muc dich hoc tap.
Xem chi tiet tai file `LICENSE`.

---

Last Updated: March 2026
