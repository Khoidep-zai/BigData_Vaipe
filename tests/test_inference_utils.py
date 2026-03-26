from pathlib import Path

import numpy as np
from PIL import Image

from src.data.features import compute_image_statistics
from src.inference.inference import _compare_colors, _compare_size, _compare_texture


def _create_image(path: Path, color=(128, 128, 128), size=(100, 50)):
    img = Image.new("RGB", size, color=color)
    img.save(path)
    return img


def test_color_size_texture_scores(tmp_path: Path):
    img1_path = tmp_path / "a.jpg"
    img2_path = tmp_path / "b.jpg"
    img1 = _create_image(img1_path, color=(120, 120, 120))
    img2 = _create_image(img2_path, color=(130, 130, 130))

    color_score = _compare_colors(img1, img2)
    size_score = _compare_size(img1, img2)
    texture_score = _compare_texture(img1, img2)

    assert 0.0 <= color_score <= 1.0
    assert 0.0 <= size_score <= 1.0
    assert 0.0 <= texture_score <= 1.0
