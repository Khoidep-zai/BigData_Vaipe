# AGENTS.md — THUOC Project
> Đọc file này TRƯỚC KHI viết bất kỳ dòng code nào.
> Compatible: Claude Code · Cursor · Windsurf · GitHub Copilot Pro · Google Antigravity · Aider

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

# Chuẩn hóa cấu trúc thư mục (chỉ chạy 1 lần sau khi có data)
python src/data/build_epillid_data.py

# Full pipeline: train → evaluate → ensemble → report  [LỆNH CHÍNH]
python run_all.py

# Train 1 model đơn lẻ
python train_cli.py --mode single --model resnet50 --epochs 28
python train_cli.py --mode single --model efficientnet_b0 --epochs 28
python train_cli.py --mode single --model vit_b_16 --epochs 32

# Hyperparameter tuning tự động nhiều vòng
python train_cli.py --mode optimize --rounds 3 --epochs 12

# Chỉ evaluate (đã có checkpoint, không train lại)
python run_all.py --compare-only

# Dùng CPU
python run_all.py --device cpu

# Smoke test / debug (không dùng để đánh giá chất lượng)
python train_cli.py --mode single --model resnet50 --epochs 2 --batch-size 4

# Unit tests
python -m pytest tests/ -q
```

---

## KIẾN TRÚC HỆ THỐNG

### Pipeline end-to-end

```
[DATA]          data_aligned/ (ưu tiên) · cấu trúc: split/class/*.jpg
                Bắt buộc: train/ · val/ · test/ — class folders phải GIỐNG NHAU
        │
        ▼
[DATA MODULE]   PillImageDataset → (image_tensor, class_idx, image_path)
                Train: augmentation ON  |  Val/Test: deterministic
        │
        ▼
[MODEL FACTORY] create_model(name) → ResNet50 / EfficientNet-B0 / ViT-B/16
                Pretrained ImageNet → fallback random init nếu offline
        │
        ▼
[TRAINING]      Stage 1: freeze backbone → Stage 2: unfreeze full model
                Loss: CrossEntropyLoss + Label Smoothing + Mixup
                EMA · TTA · Gradient Clipping · Early Stopping · AdamW · ReduceLROnPlateau
                Save: *_epillid_best.pt khi val_acc tốt nhất
        │
        ▼
[EVALUATION]    accuracy · macro-F1 · confusion matrix / model
        │
        ▼
[ENSEMBLE]      Weighted Soft-Voting: P_ens(c) = Σ(w_m × P_m(c)) / Σ(w_m)
        │
        ▼
[OUTPUT]        models/ → checkpoints · metrics · history · curves · reports
```

### Tại sao 3 model bổ trợ nhau

| Model | Điểm mạnh | Cơ chế |
|-------|----------|--------|
| **ResNet50** | Đặc trưng cục bộ: vạch chia, cạnh, màu | Residual blocks, tránh vanishing gradient |
| **EfficientNet-B0** | Chi tiết nhỏ, dị thường bề mặt | Compound Scaling depth/width/resolution |
| **ViT-B/16** | Global context, text khắc trên thuốc | Self-attention trên patch 16×16 |

---

## CẤU TRÚC THƯ MỤC

```
THUOC/
├── run_all.py                    # Entrypoint chính
├── train_cli.py                  # CLI: single / optimize / smoke test
├── requirements.txt
├── AGENTS.md                     # File này
│
├── src/
│   ├── data/
│   │   ├── features.py           # PillImageDataset · transforms · class_to_idx
│   │   ├── metadata.py           # Parse CSV · map class→metadata
│   │   └── build_epillid_data.py # Align raw → train/val/test/class/
│   ├── models/
│   │   ├── resnet50.py
│   │   ├── efficientnet_b0.py
│   │   ├── vit_b_16.py
│   │   └── model_factory.py      # ⭐ Factory DUY NHẤT để tạo model / load checkpoint
│   ├── training/
│   │   └── train.py              # Train loop · EMA · TTA · grad clip · early stop
│   ├── orchestration/
│   │   └── pipeline.py           # data → factory → train → evaluate → ensemble
│   ├── evaluation/
│   │   └── evaluate_report.py    # Metrics · CSV · chart · confusion matrix
│   ├── inference/
│   │   └── inference.py          # Feature extraction · color/size/texture · rule-based gating
│   └── learning/
│       └── self_learning.py      # Log hard examples · đề xuất retrain
│
├── Review/
│   ├── review_terminal.py        # Phân tích loss → đề xuất config vòng sau
│   └── optimal_configs.py        # 💎 NGUỒN SỰ THẬT: OPTIMAL_CONFIGS + TUNING_CANDIDATES
│
├── models/                       # Artifacts đầu ra (không commit .pt)
│   ├── *_epillid_best.pt
│   ├── *.metrics.json · *.history.json
│   ├── *_training_curves.png
│   ├── evaluation_summary.csv
│   ├── evaluation_comparison.png
│   └── reports/latest/
│       ├── evaluation_summary.json
│       └── confusion_matrix_*.png
│
├── data_aligned/                 # Dataset chuẩn hóa (không commit)
├── data/                         # Dataset gốc (không commit)
├── demo_images/
└── tests/
    ├── test_features.py
    ├── test_inference_utils.py
    └── test_metadata.py
```

---

## CONTRACTS — VI PHẠM = PHÁ VỠ PIPELINE

### Contract 1 — Dataset structure
```
data_aligned/train|val|test/class_name/*.jpg
Class folders phải GIỐNG NHAU ở cả 3 split → nếu không, class mapping sai
Fix: chạy lại build_epillid_data.py
```

### Contract 2 — Checkpoint naming
```
resnet50_epillid_best.pt
efficientnet_b0_epillid_best.pt
vit_b_16_epillid_best.pt
# Pipeline đọc theo pattern này — không đặt tên khác
```

### Contract 3 — Dataset tuple output
```python
(image_tensor, class_idx, image_path)  # thứ tự không được thay đổi
```

### Contract 4 — Class mapping safety (QUAN TRỌNG NHẤT)
```python
# LUÔN dùng helper này — KHÔNG tự load state_dict thẳng
from src.models.model_factory import load_checkpoint_class_to_idx
model, class_to_idx = load_checkpoint_class_to_idx(
    model_name="resnet50",
    checkpoint_path="models/resnet50_epillid_best.pt"
)
```

---

## HYPERPARAMETERS MẶC ĐỊNH

> Nguồn sự thật: `Review/optimal_configs.py` → `OPTIMAL_CONFIGS`  
> Không thay đổi mà không ghi lý do vào DECISIONS LOG.

| Model | lr | weight_decay | label_smoothing | mixup_alpha | epochs | patience |
|---|---:|---:|---:|---:|---:|---:|
| ResNet50 | 6e-5 | 1.2e-3 | 0.16 | 0.35 | 28 | 6 |
| EfficientNet-B0 | 7e-5 | 1e-3 | 0.15 | 0.33 | 28 | 6 |
| ViT-B/16 | 5e-5 | 1.4e-3 | 0.20 | 0.42 | 32 | 7 |

---

## PATTERNS BẮT BUỘC

```python
# Tạo model — luôn qua factory
from src.models.model_factory import create_model
model = create_model("resnet50", num_classes=8,
                     checkpoint_path="models/resnet50_epillid_best.pt")

# Dataset — luôn dùng PillImageDataset
from src.data.features import PillImageDataset, get_transforms
dataset = PillImageDataset(root="data_aligned/train", transform=get_transforms("train"))

# Config — luôn từ optimal_configs.py
from Review.optimal_configs import OPTIMAL_CONFIGS
cfg = OPTIMAL_CONFIGS["vit_b_16"]

# Inference đơn ảnh
from src.inference.inference import predict_image
result = predict_image(image_path="demo.jpg", model_name="resnet50",
                       checkpoint_path="models/resnet50_epillid_best.pt")
```

---

## EVALUATION TARGETS

```
Mỗi model đơn lẻ  : accuracy > 85%,  macro-F1 > 0.83
Ensemble           : accuracy > 90%,  macro-F1 > 0.88
```

---

## ── SKILL: GITHUB COPILOT PRO ──────────────────────────────────────────────

> Cấu hình để Copilot hiểu đúng codebase THUOC ngay từ đầu.

### Bước 1 — Tạo file `.github/copilot-instructions.md`
```markdown
# THUOC — Copilot Instructions

## Stack
Python 3.10+ · PyTorch · torchvision · FastAPI (nếu có API)

## Rules
- Luôn tạo model qua `src/models/model_factory.create_model()`, không khởi tạo trực tiếp
- Luôn dùng `PillImageDataset` từ `src/data/features.py`, không dùng `ImageFolder`
- Lấy hyperparameter từ `Review/optimal_configs.OPTIMAL_CONFIGS`, không hardcode
- Checkpoint phải đặt tên `<model>_epillid_best.pt`
- Khi load checkpoint để infer: dùng `load_checkpoint_class_to_idx`
- Không viết train logic trong `pipeline.py`, không update weights trong `evaluate_report.py`

## Structure
- src/models/      → kiến trúc model
- src/training/    → train loop (EMA, TTA, mixup, early stop)
- src/data/        → dataset, transform
- src/inference/   → predict + so sánh ảnh
- src/evaluation/  → metrics, report
- Review/          → hyperparameter configs
```

### Bước 2 — Prompts mẫu cho Copilot Chat

```
# Thêm model mới
"Tạo file src/models/convnext_tiny.py theo đúng interface của resnet50.py.
 Thêm 'convnext_tiny' vào model_factory.py. Num_classes=8."

# Debug accuracy thấp
"val_acc của resnet50 đang 78%, thấp hơn target 85%.
 Phân tích train.py và optimal_configs.py, đề xuất điều chỉnh
 weight_decay và mixup_alpha. Không thay đổi kiến trúc model."

# Thêm metric mới
"Thêm per-class F1 vào evaluate_report.py.
 Lưu kết quả vào models/reports/latest/per_class_f1.json.
 Không sửa evaluation_summary.csv hiện tại."

# Viết unit test
"Viết pytest test cho PillImageDataset trong tests/test_features.py.
 Mock file I/O. Test: output tuple đúng thứ tự, transform train khác eval,
 class_to_idx nhất quán giữa split."

# Tối ưu inference speed
"Tối ưu inference.py để batch predict nhiều ảnh cùng lúc.
 Giữ nguyên interface predict_image() hiện tại.
 Thêm predict_batch(image_paths: list[str]) → list[dict]."
```

### Bước 3 — Copilot Agent Mode (VS Code)

```
# Mở Agent Mode: Ctrl+Shift+I (hoặc Cmd+Shift+I trên Mac)

Task tốt cho Agent Mode:
✅ "Train 1 model và generate report hoàn chỉnh"
✅ "Refactor toàn bộ src/evaluation/ để hỗ trợ thêm metric AUROC"
✅ "Viết đầy đủ test suite cho src/data/ với coverage > 80%"

Task nên dùng Inline Chat (Ctrl+I) thay vì Agent:
⚡ Fix 1 bug cụ thể
⚡ Giải thích đoạn code
⚡ Refactor 1 function
```

### Bước 4 — Tips tối ưu token với Copilot Pro

```
1. Mở đúng file liên quan trước khi hỏi
   → Copilot dùng open tabs làm context — mở model_factory.py khi hỏi về model

2. Dùng #file reference trong chat
   → "Dựa vào #file:src/training/train.py, giải thích EMA được implement thế nào?"

3. Đặt câu hỏi theo layer
   → "Trong layer evaluation, tôi muốn thêm X" — Copilot không lan sang layer khác

4. Sau khi Copilot tạo code, luôn verify:
   → Checkpoint naming đúng convention chưa?
   → Có bypass factory không?
   → Có hardcode HP không?
```

---

## ── SKILL: GOOGLE ANTIGRAVITY ──────────────────────────────────────────────

> Antigravity là agent-first IDE của Google (ra mắt Nov 2025, fork từ VS Code).
> Hỗ trợ Gemini 3 Pro · Claude Sonnet 4.6 · GPT-OSS.
> 2 view chính: **Editor View** (giống VS Code + AI sidebar) và **Manager View** (multi-agent).

### Bước 1 — Cấu hình Knowledge Base (bắt buộc làm trước)

Trong Antigravity → **Knowledge Base** → Add Entry:

```
Entry 1 — Project Context
Title: THUOC Architecture
Content: [paste toàn bộ section KIẾN TRÚC HỆ THỐNG + CONTRACTS ở trên]

Entry 2 — Coding Rules
Title: THUOC Coding Rules
Content: [paste toàn bộ section PATTERNS BẮT BUỘC + BOUNDARIES ở dưới]

Entry 3 — Hyperparameters
Title: THUOC Default HP
Content: [paste bảng hyperparameter ở trên]
```

> Lý do: Antigravity agent tự kéo từ Knowledge Base khi làm task — không cần nhắc lại mỗi session.

### Bước 2 — Chọn model đúng theo task

| Task | Model nên dùng | Lý do |
|------|---------------|-------|
| Train loop, pipeline, orchestration | **Claude Sonnet 4.6** | Hiểu context dài, code phức tạp |
| Debug lỗi cụ thể, quick fix | **Gemini 3 Flash** | Nhanh, tiết kiệm quota |
| Thiết kế kiến trúc mới, refactor lớn | **Gemini 3 Pro** | Reasoning sâu |
| Giải thích algorithm (mixup, EMA, TTA) | **Claude Sonnet 4.6** | Giải thích tốt hơn |

### Bước 3 — Editor View: inline tasks thường dùng

```
# Highlight code trong train.py → Cmd+K (inline command):
"Thêm gradient clipping vào đây, max_norm=1.0, theo đúng style hiện tại"

"Refactor early stopping này thành class riêng trong cùng file,
 giữ nguyên interface gọi từ bên ngoài"

"Viết docstring cho function này theo Google style"

# Chat panel — hỏi về codebase:
"Giải thích luồng dữ liệu từ PillImageDataset đến khi model nhận batch"
"Tại sao dùng TTA trong validation mà không phải test?"
```

### Bước 4 — Manager View: multi-agent cho tasks lớn

```
# Dispatch 3 agent song song — ví dụ khi refactor lớn:

Agent 1 — Data Agent
Task: "Trong src/data/features.py, thêm support cho ảnh RGBA
       (convert sang RGB trước khi transform).
       Chạy tests/test_features.py để verify."

Agent 2 — Model Agent
Task: "Trong src/models/, kiểm tra cả 3 file model có dùng
       đúng num_classes từ tham số không.
       Report file nào có vấn đề."

Agent 3 — Test Agent
Task: "Viết integration test trong tests/test_pipeline.py:
       chạy full pipeline với 2 epochs, verify output file
       evaluation_summary.csv tồn tại và có đủ 3 model."

# Lưu ý: mỗi agent làm việc độc lập — đặt task không overlap nhau
```

### Bước 5 — Terminal Command Policy (bắt buộc cấu hình)

```
Vào Settings → Terminal Command Auto Execution:

ALLOW LIST (agent tự chạy không cần hỏi):
  python -m pytest tests/ *
  python train_cli.py --mode single *
  python run_all.py --compare-only
  pip install *
  ruff check *
  ruff format *

DENY LIST (agent phải hỏi trước khi chạy):
  python run_all.py          # train full — tốn thời gian
  rm -rf *
  git push *
  python src/data/build_epillid_data.py   # sửa cấu trúc data

# Lý do: tránh agent vô tình train lại toàn bộ hoặc xóa artifacts
```

### Bước 6 — Workflow tiêu biểu với Antigravity

```
Ví dụ: Thêm model ConvNeXt-Tiny vào pipeline

1. Editor View — hỏi context trước:
   "Đọc src/models/resnet50.py và model_factory.py.
    Mô tả interface tôi cần implement để thêm model mới."

2. Editor View — tạo file:
   "Tạo src/models/convnext_tiny.py theo đúng interface vừa mô tả.
    Thêm entry 'convnext_tiny' vào model_factory.py."

3. Manager View — verify song song:
   Agent 1: "Chạy python train_cli.py --mode single --model convnext_tiny --epochs 2"
   Agent 2: "Chạy python -m pytest tests/ -q và report kết quả"

4. Editor View — nếu test fail:
   "Test X fail với error Y. Fix trong [file cụ thể], không sửa file khác."
```

### Bước 7 — Tránh các vấn đề thường gặp với Antigravity

```
⚠️ Context window drift (sau 10-15 message, agent "quên" rules):
   → Bắt đầu session mới + re-prime: "Đọc AGENTS.md trước khi làm task này"

⚠️ Agent tự ý thay đổi file ngoài scope:
   → Luôn ghi rõ "Chỉ sửa [file cụ thể], không động vào file khác"

⚠️ Gemini 3 Pro hallucinate tên thư viện PyTorch:
   → Switch sang Claude Sonnet 4.6 cho các task liên quan torch API

⚠️ Agent chạy python run_all.py đầy đủ gây tốn thời gian:
   → Đã có trong DENY LIST — agent phải hỏi trước
```

---

## BOUNDARIES

### AI KHÔNG ĐƯỢC
```
❌ Bypass model_factory để khởi tạo model trực tiếp
❌ Tự viết DataLoader mà không dùng PillImageDataset
❌ Hardcode hyperparameter — phải qua OPTIMAL_CONFIGS
❌ Save checkpoint với tên không theo *_epillid_best.pt
❌ Tự load state_dict thẳng — phải dùng load_checkpoint_class_to_idx
❌ Thêm model mới mà không tạo file riêng trong src/models/
❌ Viết train logic trong pipeline.py
❌ Update weights trong evaluate_report.py
❌ Sửa augmentation ngoài src/data/features.py
❌ Commit data/, data_aligned/, models/*.pt lên git
❌ Thay đổi HP mà không ghi lý do vào DECISIONS LOG
```

### AI LUÔN PHẢI
```
✅ Chạy pytest -q trước khi commit thay đổi logic
✅ Verify class folders nhất quán giữa 3 split trước khi train
✅ Smoke test sau khi sửa transform: --epochs 2 --batch-size 4
✅ Sau khi train: kiểm tra evaluation_summary.csv có đủ 3 model
✅ Khi thêm class mới: cập nhật num_classes ở CẢ 3 model + factory
✅ Đọc file tương tự trong src/ trước khi tạo file mới
✅ Ghi lý do vào DECISIONS LOG khi thay đổi HP hoặc kiến trúc
```

---

## XỬ LÝ LỖI THƯỜNG GẶP

| Triệu chứng | Nguyên nhân | Fix |
|------------|------------|-----|
| Class mismatch / mapping sai | train/val/test khác class folders | Chạy lại `build_epillid_data.py` |
| `CUDA unavailable` | Không có GPU | `--device cpu` |
| Accuracy < 80% | Thiếu data hoặc LR sai | Giảm LR · `--mode optimize` |
| Pretrained download fail | Offline | Factory fallback random init (xem log) |
| Thiếu report | Pipeline interrupt | Chạy lại `python run_all.py` |
| `google.colab` warning | Chạy notebook ngoài Colab | Bỏ qua, không phải lỗi src/ |
| Git reject .pt | File > 100MB | `models/*.pt` vào `.gitignore` |
| Val loss plateau | LR quá cao | Giảm LR 50% · tăng patience |
| Overfitting | Regularization yếu | Tăng `weight_decay` · `mixup_alpha` |
| Inference sai class | Class index không đồng bộ | Dùng `load_checkpoint_class_to_idx` |
| Antigravity "quên" rules | Context drift sau nhiều message | Session mới + đọc lại AGENTS.md |

---

## ARTIFACTS PHẢI CÓ KHI NỘP

```bash
# Verify nhanh
python -m pytest tests/ -q
python run_all.py --compare-only
ls models/*.pt | wc -l                                   # → 3
ls models/reports/latest/confusion_matrix_*.png | wc -l  # → 3
git status                                               # phải sạch
```

```
models/
├── resnet50_epillid_best.pt
├── efficientnet_b0_epillid_best.pt
├── vit_b_16_epillid_best.pt
├── *.metrics.json + *.history.json  (3 file mỗi loại)
├── *_training_curves.png  (3 file)
├── evaluation_summary.csv
├── evaluation_comparison.png
└── reports/latest/
    ├── evaluation_summary.json
    └── confusion_matrix_*.png  (3 file)
```

---

## KHI GẶP AMBIGUITY

```
Cần clarify trước khi implement:
1. [câu hỏi về requirement]
2. [câu hỏi về edge case]

Assumption mặc định:
- Dùng data_aligned/ (ưu tiên hơn data/)
- Chạy cả 3 model
- Device: tự detect GPU, fallback CPU
- Checkpoint naming: *_epillid_best.pt
```

---

## DECISIONS LOG

| Ngày | Thay đổi | Lý do | Kết quả |
|------|---------|-------|---------|
| Mar 2026 | HP mặc định (bảng trên) | Tuned qua optimize mode | Baseline |
| Mar 2026 | HP: tăng regularization (ls=0.16, mixup=0.35, wd=1.2e-3), giảm lr=6e-5, epochs=28 | Giảm train/val gap, tăng tốc hội tụ | Pending verify |
| Mar 2026 | Model heads: thêm Dropout(0.3) trước Linear cuối cho cả 3 model | Giảm overfit lớp classifier | Pending verify |
| Mar 2026 | Augmentation: +GaussianBlur, +RandomAdjustSharpness, mở rộng RandomErasing | Robust hơn với ảnh thuốc mờ/sắc | Pending verify |
| Mar 2026 | image_to_numeric_vector: +HSV histogram(24d), +edge density, +quadrant lum | Vector 295-dim phong phú hơn 264-dim | Pending verify |
| Mar 2026 | train.py: warmup 2→3 epoch, eta_min 0.08→0.05, warmup_start 0.35→0.25 | Ổn định head trước khi unfreeze backbone | Pending verify |

---

*AGENTS.md v2.4 — THUOC edition, March 2026*  
*Symlink cho Claude Code: `ln -s AGENTS.md CLAUDE.md`*  
*Cập nhật khi: thêm model/class · thay đổi pipeline · đổi stack*