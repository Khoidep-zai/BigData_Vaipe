from pathlib import Path

from PIL import Image
import numpy as np
import torch

from src.data.features import (
    PillImageDataset,
    build_transforms,
    compute_image_statistics,
    focus_on_object,
    image_to_numeric_vector,
    image_vector_feature_names,
)


def _make_split_sample(split_dir: Path, class_name: str, idx: int, color: tuple[int, int, int]) -> None:
    cls_dir = split_dir / class_name
    cls_dir.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (96, 64), color=color)
    img.save(cls_dir / f"sample_{idx}.jpg")


def test_build_transforms_train_and_eval():
    # Kiểm tra tính hợp lệ: đảm bảo cả 2 quy trình biến đổi ảnh (train và eval) đều khởi tạo thành công.
    t_train = build_transforms(train=True)
    t_eval = build_transforms(train=False)
    assert t_train is not None
    assert t_eval is not None


def test_build_transforms_lab6_profile_with_grayscale():
    t_train_lab6 = build_transforms(train=True, profile="lab6_stable", grayscale_prob=0.25)
    t_eval_lab6 = build_transforms(train=False, profile="lab6_stable", grayscale_prob=0.25)
    assert t_train_lab6 is not None
    # Eval path stays deterministic regardless of train profile choice.
    assert t_eval_lab6 is not None


def test_compute_image_statistics_shape(tmp_path: Path):
    # Tạo ảnh giả lập để kiểm tra xem hàm tính thống kê có trả về đúng các trường dữ liệu cần thiết không.
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (100, 50), color=(128, 64, 32))
    img.save(img_path)

    stats = compute_image_statistics(img)
    assert "mean_r" in stats and "mean_g" in stats and "mean_b" in stats
    # vì có center crop nên kích thước sau thống kê có thể thay đổi,
    # chỉ cần đảm bảo hợp lệ và aspect_ratio dương
    assert stats["height"] > 0
    assert stats["width"] > 0
    assert stats["aspect_ratio"] > 0


def test_focus_on_object_center_crop():
    # Kiểm tra hàm cắt ảnh: đảm bảo cắt đúng vùng trung tâm theo tỷ lệ cạnh ngắn nhất.
    img = Image.new("RGB", (200, 100), color=(128, 64, 32))
    cropped = focus_on_object(img, scale=0.5)
    # cạnh ngắn = 100 -> sau crop còn 50
    assert cropped.size[0] == 50 or cropped.size[1] == 50


def test_image_to_numeric_vector_shape_and_values():
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    vec = image_to_numeric_vector(img, thumbnail_size=8)
    names = image_vector_feature_names(thumbnail_size=8)

    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32
    assert vec.shape[0] == len(names)
    assert np.isfinite(vec).all()


def test_pill_image_dataset_returns_expected_tuple_order(tmp_path: Path):
    train_root = tmp_path / "train"
    _make_split_sample(train_root, "class_a", 0, (120, 30, 10))

    dataset = PillImageDataset(root=str(train_root), transform=build_transforms(train=False))
    image_tensor, class_idx, image_path = dataset[0]

    assert isinstance(image_tensor, torch.Tensor)
    assert isinstance(class_idx, int)
    assert isinstance(image_path, str)
    assert image_tensor.ndim == 3 and image_tensor.shape[0] == 3
    assert image_path.endswith("sample_0.jpg")


def test_pill_image_dataset_class_to_idx_consistent_between_splits(tmp_path: Path):
    train_root = tmp_path / "train"
    val_root = tmp_path / "val"
    classes = ["class_alpha", "class_beta"]

    for idx, cls_name in enumerate(classes):
        _make_split_sample(train_root, cls_name, idx, (30 * (idx + 1), 50, 100))
        _make_split_sample(val_root, cls_name, idx, (30 * (idx + 1), 80, 140))

    train_ds = PillImageDataset(root=str(train_root), transform=build_transforms(train=True))
    val_ds = PillImageDataset(
        root=str(val_root),
        transform=build_transforms(train=False),
        class_to_idx=train_ds.class_to_idx,
    )

    assert train_ds.class_to_idx == val_ds.class_to_idx
    assert set(train_ds.class_to_idx.keys()) == set(classes)


def test_train_and_eval_transforms_produce_same_shape(tmp_path: Path):
    train_root = tmp_path / "train"
    _make_split_sample(train_root, "class_a", 0, (90, 100, 110))

    dataset_train = PillImageDataset(root=str(train_root), transform=build_transforms(train=True))
    dataset_eval = PillImageDataset(root=str(train_root), transform=build_transforms(train=False))

    train_tensor, _, _ = dataset_train[0]
    eval_tensor, _, _ = dataset_eval[0]

    # Train có augment ngẫu nhiên nhưng cả 2 nhánh đều phải trả tensor 3x224x224.
    assert train_tensor.shape == eval_tensor.shape == (3, 224, 224)

