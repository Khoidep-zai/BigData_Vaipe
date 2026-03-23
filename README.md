# THUOC - Hướng Dẫn Dễ Hiểu Cho Người Mới

THUOC là đồ án phân loại viên thuốc từ ảnh. Bạn không cần hiểu sâu về AI vẫn có thể chạy được toàn bộ quy trình nếu làm theo đúng các bước trong tài liệu này.

## Bạn Sẽ Nhận Được Gì Sau Khi Chạy?

Sau khi chạy xong, hệ thống sẽ tự tạo:
1. 3 mô hình đã huấn luyện (ResNet50, EfficientNet-B0, ViT-B/16).
2. Bảng kết quả so sánh Accuracy và Macro-F1.
3. Biểu đồ so sánh mô hình.
4. Confusion matrix cho từng mô hình và ensemble.

## Kết Quả Hiện Tại

Mô hình đã được huấn luyện thành công với 5 epoch (early stop do gap train/val vượt ngưỡng). Kết quả đánh giá trên test set:

| Model | Accuracy | Macro-F1 | Best Epoch | Train/Val Gap |
|---|---:|---:|---:|---:|
| ResNet50 | 20.0% | 0.1333 | Epoch 4 | 0.30 |
| EfficientNet-B0 | 30.0% | 0.1958 | Epoch 5 | 0.40 |
| ViT-B/16 | 20.0% | 0.1333 | Epoch 5 | 0.18 |

*Ghi chú: Kết quả thấp là do dataset nhỏ (10 mẫu test) và validation set hạn chế. Hãy tăng kích cỡ dữ liệu hoặc điều chỉnh hyperparameter để cải thiện.*

## Chạy Nhanh Trong 5 Phút

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

Đây là lệnh quan trọng nhất. Lệnh này sẽ train, evaluate và xuất report tự động.

## Chạy Trên Google Colab

Bạn có notebook sẵn sàng cho Colab tại: `THUOC_Colab_Train_Evaluate.ipynb`

### Cách dùng:
1. Mở file notebook này trên Google Colab.
2. Chạy Cell 1 và sửa `REPO_URL` thành repo GitHub thật của bạn.
3. Nếu dữ liệu nằm trong Google Drive, giữ `USE_DRIVE_DATA = True` và sửa `DRIVE_DATA_ROOT` đúng.
4. Chạy lần lượt từ Cell 2 đến Cell 9.

**Lợi ích:**
- Train trên GPU miễn phí của Colab (nhanh hơn CPU 10-50 lần).
- Tự động sinh đầy đủ checkpoint, metrics, confusion matrix.
- Đóng gói output thành ZIP để tải về hoặc lưu lại Drive.

## Lệnh Quan Trọng Nhất

| Mục tiêu | Lệnh |
|---|---|
| Chạy toàn bộ 3 mô hình | python run_all.py |
| Chạy 1 mô hình | python run_all.py --model resnet50 |
| Chỉ evaluate model đã train | python run_all.py --compare-only |
| Dùng CPU (nếu không có GPU) | python run_all.py --device cpu |
| Chỉ định data thủ công | python run_all.py --data-dir data_aligned |

## Kết Quả Nằm Ở Đâu?

Sau khi chạy, bạn xem kết quả trong thư mục models:

```text
models/
  resnet50_epillid_best.pt
  efficientnet_b0_epillid_best.pt
  vit_b_16_epillid_best.pt

  resnet50_epillid_best.metrics.json
  efficientnet_b0_epillid_best.metrics.json
  vit_b_16_epillid_best.metrics.json

  resnet50_epillid_history.json
  efficientnet_b0_epillid_history.json
  vit_b_16_epillid_history.json

  resnet50_training_curves.png
  efficientnet_b0_training_curves.png
  vit_b_16_training_curves.png

  evaluation_summary.csv
  evaluation_comparison.png

  reports/latest/
    evaluation_summary.csv
    evaluation_summary.json
    evaluation_comparison.png
    tuning_summary.json
    confusion_matrix_resnet50.png
    confusion_matrix_efficientnet_b0.png
    confusion_matrix_vit_b_16.png
    confusion_matrix_ensemble_weighted.png
```

Nếu bạn cần file để nộp đồ án, phần quan trọng nhất là:
1. models/evaluation_summary.csv
2. models/evaluation_comparison.png
3. models/reports/latest/evaluation_summary.json
4. models/reports/latest/confusion_matrix_*.png
5. models/*_epillid_best.pt (3 checkpoint)
6. models/*_training_curves.png (3 biểu đồ)

## Sơ Đồ Dòng Chạy (Đơn Giản)

```mermaid
flowchart LR
  A[Du lieu: data_aligned hoac data] --> B[python run_all.py]
  B --> C[Train 3 model]
  C --> D[Evaluate]
  D --> E[Xuat CSV JSON PNG]
```

## Khi Nào Dùng train_cli.py?

run_all.py đã đủ cho hầu hết trường hợp. Chỉ dùng train_cli.py khi bạn muốn can thiệp sâu hơn.

```bash
# Full pipeline (chi tiet)
python train_cli.py --mode all --epochs 28 --batch-size 16

# Train 1 model
python train_cli.py --mode single --model resnet50 --epochs 28

# Train + review nhieu vong
python train_cli.py --mode optimize --rounds 3 --epochs 12
```

## Hyperparameter Mặc Định Hiện Tại

Cấu hình này lấy trực tiếp từ optimal_configs.py:

| Model | lr | weight_decay | label_smoothing | mixup_alpha | epochs | early_stop_patience |
|---|---:|---:|---:|---:|---:|---:|
| ResNet50 | 6e-5 | 1.2e-3 | 0.16 | 0.35 | 28 | 6 |
| EfficientNet-B0 | 7e-5 | 1e-3 | 0.15 | 0.33 | 28 | 6 |
| ViT-B/16 | 5e-5 | 1.4e-3 | 0.20 | 0.42 | 32 | 7 |

## File Quan Trọng Để Nộp Đồ Án

Chuẩn bị trước khi nộp:

1. **Source code**: Thư mục `src/` đầy đủ với tất cả module.
2. **Checkpoint**: 3 file `*_epillid_best.pt` trong `models/`.
3. **Kết quả evaluate**: `models/evaluation_summary.csv` + `models/evaluation_summary.json`.
4. **Biểu đồ so sánh**: `models/evaluation_comparison.png`.
5. **Training curves**: 3 file `*_training_curves.png` trong `models/`.
6. **Confusion matrix**: Tất cả file `confusion_matrix_*.png` trong `models/reports/latest/`.
7. **README.md** (file này).

## Cấu Trúc Thư Mục (Tóm Tắt)

```text
THUOC/
  run_all.py
  train_cli.py
  review_terminal.py
  optimal_configs.py
  requirements.txt
  THUOC_Colab_Train_Evaluate.ipynb
  src/
  data/ hoặc data_aligned/
  models/
  tests/
```

## Kiểm Thử Nhanh

```bash
python -m pytest tests/ -q
```

## Lỗi Thường Gặp Và Cách Xử Lý

1. **Lỗi không thấy dữ liệu:**
Chắc chắn bạn có data_aligned hoặc data và bên trong có train/val/test.

2. **Lỗi CUDA không khả dụng:**
Chạy lại với CPU.

```bash
python run_all.py --device cpu
```

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