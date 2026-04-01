from __future__ import annotations

import os
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


IMG_SIZE = 224
VALID_EXTS = (".jpg", ".jpeg", ".png")


def _default_train_transforms() -> T.Compose:
    return T.Compose(
        [
            # Cắt ngẫu nhiên vùng ảnh rồi scale lên 224×224 → tăng đa dạng vị trí và kích thước viên thuốc.
            T.RandomResizedCrop(
                (IMG_SIZE, IMG_SIZE),
                scale=(0.68, 1.0),
                ratio=(0.85, 1.15),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.15),
            # ColorJitter mô phỏng ánh sáng và nền khác nhau khi chụp thuốc.
            T.RandomApply(
                [
                    T.ColorJitter(
                        brightness=0.15,
                        contrast=0.15,
                        saturation=0.12,
                        hue=0.03,
                    )
                ],
                p=0.7,
            ),
            T.RandomAffine(
                degrees=10,
                translate=(0.08, 0.08),
                scale=(0.90, 1.10),
                interpolation=InterpolationMode.BILINEAR,
            ),
            T.RandomPerspective(distortion_scale=0.15, p=0.25),
            # GaussianBlur mô phỏng ảnh thuốc bị mờ do camera không focus.
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
            # RandomAdjustSharpness mô phỏng camera có chất lượng sắc nét khác nhau.
            T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2),
            T.ToTensor(),
            # RandomErasing che một phần ảnh → buộc model nhìn nhiều vùng đặc trưng hơn.
            T.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _lab6_stable_train_transforms(grayscale_prob: float = 0.0) -> T.Compose:
    # Lab6-inspired stable recipe: nhẹ hơn profile mặc định để train/val mượt hơn trên dữ liệu nhỏ.
    g_prob = float(max(0.0, min(1.0, grayscale_prob)))
    transforms: List[object] = [
        T.RandomResizedCrop(
            (IMG_SIZE, IMG_SIZE),
            scale=(0.78, 1.0),
            ratio=(0.90, 1.10),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15, interpolation=InterpolationMode.BILINEAR),
        T.RandomApply(
            [
                T.ColorJitter(
                    brightness=0.10,
                    contrast=0.10,
                    saturation=0.08,
                    hue=0.02,
                )
            ],
            p=0.35,
        ),
    ]
    if g_prob > 0:
        # Kỹ thuật vector grayscale từ Lab6 được đưa vào augment để mô hình học hình dạng ổn định hơn.
        transforms.append(T.RandomApply([T.RandomGrayscale(p=1.0)], p=g_prob))

    transforms.extend(
        [
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return T.Compose(transforms)


def build_transforms(
    train: bool = True,
    profile: str = "default",
    grayscale_prob: float = 0.0,
) -> T.Compose:
    # Giữ các bước tiền xử lý ảnh nhất quán với chuẩn của các mô hình đã huấn luyện sẵn trên ImageNet.
    profile_name = (profile or "default").strip().lower()
    if train:
        if profile_name == "lab6_stable":
            return _lab6_stable_train_transforms(grayscale_prob=grayscale_prob)
        return _default_train_transforms()
    else:
        return T.Compose(
            [
                # Đường dẫn xử lý cho đánh giá (Eval) được giữ cố định (không ngẫu nhiên) để kết quả kiểm tra luôn giống nhau mỗi lần chạy.
                T.Resize(int(IMG_SIZE * 1.15), interpolation=InterpolationMode.BICUBIC, antialias=True),
                T.CenterCrop((IMG_SIZE, IMG_SIZE)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )


# Alias giữ tương thích ngược với code gọi get_transforms().
get_transforms = build_transforms


def pil_loader(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def focus_on_object(img: Image.Image, scale: float = 0.8) -> Image.Image:
    """
    Phóng to vùng trung tâm của ảnh để tập trung hơn vào viên thuốc.

    - scale: tỷ lệ cạnh ngắn được giữ lại (0 < scale <= 1).
    - Ở đây dùng center crop đơn giản: phù hợp với dataset viên thuốc
      thường nằm gần giữa ảnh.
    """
    if not (0 < scale <= 1.0):
        return img

    w, h = img.size
    side = int(min(w, h) * scale)
    if side <= 0:
        return img

    left = (w - side) // 2
    top = (h - side) // 2
    right = left + side
    bottom = top + side
    return img.crop((left, top, right, bottom))

@dataclass
class ImageSample:
    path: str
    label: int


class PillImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform: T.Compose | None = None,
        class_to_idx: Dict[str, int] | None = None,
    ) -> None:
        # Cấu trúc thư mục mong đợi: thư_mục_gốc/<tên_lớp_thuốc>/*.jpg (hoặc png).
        self.root = root
        self.transform = transform or build_transforms(train=True)
        self.samples: List[ImageSample] = []
        self.class_to_idx = self._find_classes(class_to_idx=class_to_idx)

    def _find_classes(self, class_to_idx: Dict[str, int] | None = None) -> Dict[str, int]:
        # Tự động tìm kiếm tên các lớp thuốc dựa trên tên thư mục nếu chưa được cung cấp bảng ánh xạ.
        if class_to_idx is None:
            classes: List[str] = []
            for d in os.scandir(self.root):
                if not d.is_dir() or d.name.startswith("."):
                    continue
                has_image = any(
                    f.name.lower().endswith(VALID_EXTS)
                    for f in os.scandir(d.path)
                    if f.is_file()
                )
                if has_image:
                    classes.append(d.name)
            classes = sorted(classes)
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        # Tạo một danh sách phẳng chứa tất cả các ảnh mẫu để việc lấy dữ liệu (hàm __getitem__) nhanh và ổn định.
        for cls_name, cls_idx in class_to_idx.items():
            cls_dir = os.path.join(self.root, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(VALID_EXTS):
                    path = os.path.join(cls_dir, fname)
                    self.samples.append(ImageSample(path=path, label=cls_idx))

        return class_to_idx if class_to_idx is not None else {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        sample = self.samples[idx]
        img = pil_loader(sample.path)
        # Cắt lấy vùng trung tâm của ảnh, nơi viên thuốc thường xuất hiện, để mô hình tập trung vào đối tượng chính.
        img = focus_on_object(img, scale=0.85)
        img = self.transform(img)
        return img, sample.label, sample.path


def compute_image_statistics(img: Image.Image) -> dict:
    """Return simple statistics: mean RGB, size (sau khi focus vào vật thể)."""
    img_focused = focus_on_object(img, scale=0.85)
    arr = np.array(img_focused.convert("RGB")) / 255.0
    mean_rgb = arr.mean(axis=(0, 1))
    h, w = arr.shape[:2]
    return {
        "mean_r": float(mean_rgb[0]),
        "mean_g": float(mean_rgb[1]),
        "mean_b": float(mean_rgb[2]),
        "height": int(h),
        "width": int(w),
        "aspect_ratio": float(w / h),
    }


def _compute_hsv_histogram(img_rgb: Image.Image, bins: int = 8) -> np.ndarray:
    """Tính histogram HSV (H, S, V mỗi kênh bins giá trị) → 3*bins chiều.
    Histogram màu phong phú hơn mean RGB rất nhiều khi so sánh 2 viên thuốc."""
    hsv = img_rgb.convert("HSV")
    arr = np.array(hsv, dtype=np.float32)
    hist_parts = []
    for ch in range(3):
        channel = arr[:, :, ch]
        max_val = 360.0 if ch == 0 else 255.0
        h, _ = np.histogram(channel / max_val, bins=bins, range=(0.0, 1.0))
        total = max(h.sum(), 1)
        hist_parts.append(h.astype(np.float32) / total)
    return np.concatenate(hist_parts)


def _compute_edge_density(gray_arr: np.ndarray) -> float:
    """Tính mật độ cạnh bằng Sobel filter → phản ánh vân/chi tiết trên bề mặt thuốc."""
    from PIL import ImageFilter as _IF
    # Convert numpy gray array to PIL, apply Sobel
    gray_img = Image.fromarray((gray_arr * 255).astype(np.uint8), mode="L")
    edges = gray_img.filter(_IF.FIND_EDGES)
    edge_arr = np.array(edges, dtype=np.float32) / 255.0
    return float(edge_arr.mean())


def _compute_quadrant_luminance(gray_arr: np.ndarray) -> np.ndarray:
    """Tính luminance trung bình ở 4 góc phần tư ảnh → phân biệt thuốc hai tông màu."""
    h, w = gray_arr.shape
    mid_h, mid_w = h // 2, w // 2
    if mid_h == 0 or mid_w == 0:
        return np.zeros(4, dtype=np.float32)
    quads = [
        gray_arr[:mid_h, :mid_w],       # top-left
        gray_arr[:mid_h, mid_w:],        # top-right
        gray_arr[mid_h:, :mid_w],        # bottom-left
        gray_arr[mid_h:, mid_w:],        # bottom-right
    ]
    return np.array([float(q.mean()) for q in quads], dtype=np.float32)


def image_to_numeric_vector(img: Image.Image, thumbnail_size: int = 16) -> np.ndarray:
    """
    Chuyen anh thanh vector so de phuc vu phan tich/thuc nghiem.

    Vector v2 gom (~295 chieu voi thumbnail_size=16, bins=8):
    - 8 gia tri thong ke co ban (mean/std RGB, texture std, aspect ratio)
    - 24 gia tri histogram HSV (8 bins x 3 kenh H, S, V)
    - 1 gia tri edge density (mat do canh Sobel)
    - 4 gia tri quadrant luminance (do sang 4 goc)
    - 2 gia tri brightness + contrast
    - Anh grayscale thumbnail duoc flatten (thumbnail_size * thumbnail_size)
    """
    img_focused = focus_on_object(img, scale=0.85)
    rgb_arr = np.array(img_focused.convert("RGB"), dtype=np.float32) / 255.0
    gray_arr = np.array(img_focused.convert("L"), dtype=np.float32) / 255.0

    mean_rgb = rgb_arr.mean(axis=(0, 1))
    std_rgb = rgb_arr.std(axis=(0, 1))
    h, w = gray_arr.shape
    texture_std = float(gray_arr.std())
    aspect_ratio = float(w / max(h, 1))

    # Thống kê cơ bản (8 chiều)
    stats_basic = np.array(
        [
            float(mean_rgb[0]),
            float(mean_rgb[1]),
            float(mean_rgb[2]),
            float(std_rgb[0]),
            float(std_rgb[1]),
            float(std_rgb[2]),
            texture_std,
            aspect_ratio,
        ],
        dtype=np.float32,
    )

    # Histogram HSV (24 chiều) → nắm bắt phân phối màu chi tiết
    hsv_hist = _compute_hsv_histogram(img_focused, bins=8)

    # Edge density (1 chiều) → phản ánh chi tiết bề mặt
    edge_density = np.asarray([_compute_edge_density(gray_arr)], dtype=np.float32)

    # Quadrant luminance (4 chiều) → phân biệt thuốc hai tông
    quad_lum = np.asarray(_compute_quadrant_luminance(gray_arr), dtype=np.float32)

    # Brightness và contrast bổ sung (2 chiều)
    brightness = float(gray_arr.mean())
    contrast = float(gray_arr.max() - gray_arr.min())
    bright_contrast = np.array([brightness, contrast], dtype=np.float32)

    # Thumbnail flatten (thumbnail_size^2 chiều)
    thumb = img_focused.convert("L").resize(
        (thumbnail_size, thumbnail_size),
        resample=Image.Resampling.BILINEAR,
    )
    thumb_flat = (np.array(thumb, dtype=np.float32) / 255.0).reshape(-1)

    return np.concatenate(
        [stats_basic, hsv_hist, edge_density, quad_lum, bright_contrast, thumb_flat],
        axis=0,
    ).astype(np.float32)


def image_vector_feature_names(thumbnail_size: int = 16) -> List[str]:
    names = [
        "mean_r",
        "mean_g",
        "mean_b",
        "std_r",
        "std_g",
        "std_b",
        "texture_std",
        "aspect_ratio",
    ]
    # HSV histogram bins
    for ch_name in ["h", "s", "v"]:
        for b in range(8):
            names.append(f"hist_{ch_name}_{b}")
    # Edge density
    names.append("edge_density")
    # Quadrant luminance
    for q in ["tl", "tr", "bl", "br"]:
        names.append(f"quad_lum_{q}")
    # Brightness/contrast
    names.append("brightness")
    names.append("contrast")
    # Thumbnail pixels
    names.extend([f"px_{i:03d}" for i in range(thumbnail_size * thumbnail_size)])
    return names


def export_image_vectors_csv(
    split_root: str | Path,
    output_csv: str | Path,
    thumbnail_size: int = 16,
) -> None:
    """
    Quet du lieu theo cau truc split/class/*.jpg va xuat CSV vector hoa.
    Ham nay khong can thay doi pipeline train hien tai, chi bo sung du lieu so hoa de phan tich.
    """
    split_root = Path(split_root)
    output_csv = Path(output_csv)
    rows: List[Dict[str, object]] = []
    feature_names = image_vector_feature_names(thumbnail_size=thumbnail_size)

    classes = [d.name for d in split_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    classes = sorted(classes)

    for cls_name in classes:
        cls_dir = split_root / cls_name
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.lower().endswith(VALID_EXTS):
                continue
            img_path = cls_dir / fname
            vec = image_to_numeric_vector(pil_loader(str(img_path)), thumbnail_size=thumbnail_size)

            row: Dict[str, object] = {
                "class_name": cls_name,
                "image_path": str(img_path),
            }
            for col, val in zip(feature_names, vec.tolist()):
                row[col] = float(val)
            rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["class_name", "image_path"] + feature_names
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
