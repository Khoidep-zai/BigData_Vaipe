from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


IMG_SIZE = 224
VALID_EXTS = (".jpg", ".jpeg", ".png")


def build_transforms(train: bool = True) -> T.Compose:
    # Giữ các bước tiền xử lý ảnh nhất quán với chuẩn của các mô hình đã huấn luyện sẵn trên ImageNet.
    if train:
        return T.Compose(
            [
                # Áp dụng các biến đổi hình học và màu sắc (xoay, cắt, chỉnh màu) để giúp mô hình học tốt hơn, tránh học vẹt trên tập dữ liệu nhỏ.
                T.RandomResizedCrop(
                    (IMG_SIZE, IMG_SIZE),
                    scale=(0.72, 1.0),
                    ratio=(0.88, 1.12),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.1),
                T.RandomApply(
                    [
                        T.ColorJitter(
                            brightness=0.12,
                            contrast=0.12,
                            saturation=0.10,
                            hue=0.02,
                        )
                    ],
                    p=0.7,
                ),
                T.RandomAffine(
                    degrees=8,
                    translate=(0.06, 0.06),
                    scale=(0.92, 1.08),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                T.RandomPerspective(distortion_scale=0.15, p=0.25),
                T.ToTensor(),
                T.RandomErasing(p=0.20, scale=(0.01, 0.08), ratio=(0.3, 3.3), value="random"),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
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

        return class_to_idx

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

