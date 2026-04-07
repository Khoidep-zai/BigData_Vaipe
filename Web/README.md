# 🌐 THUOC Web - Báo cáo triển khai giao diện và API

> **Nhóm nghiên cứu:** VLU.AI-Med Team - Trường Đại học Văn Lang  
> **Vai trò module:** Cầu nối giữa mô hình deep learning và người dùng cuối

---

## 📑 Mục lục
- [1. Mục tiêu module Web](#1-mục-tiêu-module-web)
- [2. Kiến trúc triển khai](#2-kiến-trúc-triển-khai)
- [3. Luồng chức năng chính](#3-luồng-chức-năng-chính)
- [4. Tối ưu hiệu năng và độ ổn định](#4-tối-ưu-hiệu-năng-và-độ-ổn-định)
- [5. Cấu trúc thư mục](#5-cấu-trúc-thư-mục)
- [6. Cài đặt và chạy](#6-cài-đặt-và-chạy)
- [7. API chi tiết](#7-api-chi-tiết)
- [8. Biến môi trường](#8-biến-môi-trường)
- [9. Kiểm thử nhanh](#9-kiểm-thử-nhanh)
- [10. Hướng phát triển](#10-hướng-phát-triển)

---

## 1. Mục tiêu module Web
THUOC Web được xây dựng để:

- Cho phép người dùng upload/chụp ảnh trực tiếp để phân loại viên thuốc.
- Kiểm tra đúng toa bằng kết quả true/false rõ ràng.
- Hiển thị tổng quan dữ liệu, checkpoint và kết quả đánh giá mới nhất.

Đây là lớp triển khai ứng dụng thực tế cho phần nghiên cứu mô hình.

---

## 2. Kiến trúc triển khai

![Kiến trúc Web THUOC](../docs/figures/web-architecture.svg)

### 2.1. Ảnh giao diện Web thực tế

![THUOC Web Desktop](../docs/figures/web-ui-desktop.png)

![THUOC Web Mobile Preview](../docs/figures/web-ui-mobile.png)

Thành phần chính:

1. Frontend: HTML/CSS/JS thuần, tối ưu thao tác nhanh.
2. Backend Flask: API health, overview, classify, check-prescription.
3. Inference layer: tải checkpoint, chạy suy luận và trả JSON.
4. Artifacts layer: đọc dữ liệu từ models/results, models/reports/latest, data/csv.

---

## 3. Luồng chức năng chính
### 3.1. Phân loại ảnh viên thuốc
1. Người dùng chọn model và ảnh viên thuốc.
2. Backend chạy infer top-k.
3. Trả về class_id, class_name, medicine_name, confidence.

### 3.2. Kiểm tra đúng toa
1. Upload ảnh toa + nhiều ảnh viên thuốc.
2. Backend batch infer toàn bộ ảnh pill.
3. So khớp với ngữ cảnh toa từ CSV.
4. Trả về:
   - analysis_true_false
   - has_out_of_prescription
   - chi tiết từng ảnh pill

---

## 4. Tối ưu hiệu năng và độ ổn định
Bản hiện tại đã tối ưu theo hướng chạy mượt hơn:

### 4.1. Backend
- Cache overview theo TTL để tránh quét dữ liệu lặp lại quá thường xuyên.
- Batch inference cho nhiều ảnh pill trong một request.
- Giới hạn số ảnh pill mỗi request (tránh nghẽn RAM/CPU).
- Chuẩn hóa lỗi API về JSON, dễ xử lý ở frontend.
- Quản lý thư mục uploads theo ngưỡng số file để tránh phình dung lượng.

### 4.2. Frontend
- Tự hủy request cũ khi người dùng gửi request mới liên tiếp.
- Disable nút submit trong lúc đang xử lý.
- Validate số lượng ảnh pill trước khi gửi.
- Làm mới overview theo chế độ force khi cần dữ liệu mới tức thời.

### 4.3. Benchmark độ trễ API (local)

Nguồn số liệu:

- ../docs/benchmarks/inference_benchmark.json
- ../docs/benchmarks/inference_benchmark.md

| Endpoint Scenario | Device | Mean (ms) | P95 (ms) |
|---|---|---:|---:|
| classify_single_image | cpu | 524.34 | 572.28 |
| check_prescription_two_pills | cpu | 2062.26 | 2348.78 |

Ghi chú: môi trường hiện tại chưa có CUDA nên chưa có benchmark GPU.

---

## 5. Cấu trúc thư mục

```text
Web/
├─ app.py
├─ README.md
├─ backend/
│  ├─ app.py
│  └─ __init__.py
├─ frontend/
│  ├─ templates/
│  │  └─ index.html
│  └─ static/
│     ├─ app.js
│     └─ styles.css
└─ uploads/
   ├─ pills/
   └─ prescriptions/
```

---

## 6. Cài đặt và chạy
Chạy từ thư mục project root:

```bash
pip install -r requirements.txt
python Web/app.py
```

Mặc định mở tại:
- http://127.0.0.1:5000

Chạy trực tiếp backend (tùy chọn):

```bash
python Web/backend/app.py
```

---

## 7. API chi tiết
### 7.1. GET /api/health
- Trạng thái dịch vụ và thiết bị chạy mô hình.

### 7.2. GET /api/overview
- Tổng quan dataset, mô hình, metrics.
- Hỗ trợ force refresh: /api/overview?force=1

### 7.3. POST /api/classify
form-data:
- image
- model_name
- top_k

### 7.4. POST /api/check-prescription
form-data:
- prescription_image
- pill_images (nhiều ảnh)
- model_name
- top_k
- use_annotation_lookup

---

## 8. Biến môi trường
- THUOC_WEB_DEVICE=cpu|cuda (mặc định: cpu)
- THUOC_WEB_HOST (mặc định: 127.0.0.1)
- THUOC_WEB_PORT (mặc định: 5000)
- THUOC_WEB_DEBUG=0|1 (mặc định: 0)
- THUOC_WEB_MAX_UPLOAD_MB (mặc định: 16)
- THUOC_WEB_MAX_PILL_IMAGES (mặc định: 30)
- THUOC_WEB_MAX_UPLOAD_FILES (mặc định: 500)
- THUOC_WEB_OVERVIEW_TTL_SEC (mặc định: 25)

---

## 9. Kiểm thử nhanh
Sau khi chạy server, có thể test nhanh:

1. GET /api/health để kiểm tra server hoạt động.
2. GET /api/overview?force=1 để xác nhận dữ liệu phản hồi đầy đủ.
3. Thử classify với 1 ảnh pill.
4. Thử check-prescription với 1 ảnh toa và nhiều ảnh pill.

---

## 10. Hướng phát triển
- Thêm xác thực người dùng và nhật ký truy vết request.
- Tối ưu infer bất đồng bộ cho khối lượng request lớn.
- Bổ sung dashboard theo dõi realtime metrics.
- Đóng gói Docker để triển khai nhanh tại môi trường bệnh viện/nhà thuốc.
