from pathlib import Path

from PIL import Image

from src.data.features import build_transforms, compute_image_statistics, focus_on_object


def test_build_transforms_train_and_eval():
    t_train = build_transforms(train=True)
    t_eval = build_transforms(train=False)
    assert t_train is not None
    assert t_eval is not None


def test_compute_image_statistics_shape(tmp_path: Path):
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
    img = Image.new("RGB", (200, 100), color=(128, 64, 32))
    cropped = focus_on_object(img, scale=0.5)
    # cạnh ngắn = 100 -> sau crop còn 50
    assert cropped.size[0] == 50 or cropped.size[1] == 50
