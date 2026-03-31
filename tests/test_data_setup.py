import json
from pathlib import Path
from typing import cast

import pandas as pd
from PIL import Image

from src.data.data_setup import (
    build_thuoc_data_from_vaipe,
    clean_metadata_csv_for_vectors,
    prepare_metadata_artifacts,
)


def _make_vaipe_sample(root: Path, class_id: int, idx: int) -> None:
    image_dir = root / "public_train" / "pill" / "image"
    label_dir = root / "public_train" / "pill" / "label"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    stem = f"VAIPE_P_{class_id}_{idx}"
    image_path = image_dir / f"{stem}.jpg"
    label_path = label_dir / f"{stem}.json"

    img = Image.new("RGB", (128, 128), color=(20 * class_id, 40, 120))
    img.save(image_path)

    anns = [{"x": 24, "y": 24, "w": 64, "h": 64, "label": class_id}]
    label_path.write_text(json.dumps(anns), encoding="utf-8")


def test_build_thuoc_data_from_vaipe(tmp_path: Path):
    raw_root = tmp_path / "data"
    out_root = tmp_path / "data_aligned"

    for i in range(6):
        _make_vaipe_sample(raw_root, class_id=1, idx=i)
        _make_vaipe_sample(raw_root, class_id=2, idx=i)

    summary = build_thuoc_data_from_vaipe(
        raw_root=raw_root,
        output_root=out_root,
        val_ratio=0.2,
        test_ratio=0.2,
        min_samples_per_class=3,
        seed=7,
    )

    assert summary["num_classes"] == 2

    train_classes = sorted([p.name for p in (out_root / "train").iterdir() if p.is_dir()])
    val_classes = sorted([p.name for p in (out_root / "val").iterdir() if p.is_dir()])
    test_classes = sorted([p.name for p in (out_root / "test").iterdir() if p.is_dir()])

    assert train_classes == ["class_001", "class_002"]
    assert train_classes == val_classes == test_classes

    assert any((out_root / "train" / "class_001").glob("*.jpg"))
    assert any((out_root / "val" / "class_001").glob("*.jpg"))
    assert any((out_root / "test" / "class_001").glob("*.jpg"))


def test_clean_metadata_csv_for_vectors(tmp_path: Path):
    input_csv = tmp_path / "metadata_raw.csv"
    output_csv = tmp_path / "metadata_clean.csv"

    input_csv.write_text(
        "Medicine Name,Composition,Dosage_Form,Color_For_AI,Shape_For_AI,Manufacturer,Disease_Treated_VI\n"
        "Drug A,Comp A,Tablet,Do,Tron,Maker 1,Benh A\n"
        "Drug A,Comp A,Tablet,Do,Tron,Maker 1,Benh A\n"
        ",,Capsule,Vang,Bau,Maker 2,Benh B\n",
        encoding="utf-8",
    )

    summary = clean_metadata_csv_for_vectors(input_csv, output_csv)
    out_df = cast(pd.DataFrame, pd.read_csv(output_csv, encoding="utf-8-sig"))

    assert summary["rows_in"] == 3
    assert summary["rows_out"] == 1
    assert "Manufacturer" not in out_df.columns
    assert "Medicine Name" in out_df.columns
    assert "Composition" in out_df.columns


def test_prepare_metadata_artifacts_supports_data_csv_layout(tmp_path: Path):
    data_root = tmp_path / "data"
    csv_root = data_root / "csv"
    csv_root.mkdir(parents=True, exist_ok=True)

    raw_csv = csv_root / "Medicine_Details_Deeplearning.csv"
    raw_csv.write_text(
        "Medicine Name,Composition,VAIPE2022_Class_ID,Dosage_Form,Weight,Length_mm,Width_mm,Height_mm,Color_For_AI,Shape_For_AI,Active_Ingredient_Group,Disease_Treated_VI\n"
        "Drug A,Comp A,0,Tablet,10x10x4,10,10,4,White,Round,Group A,Disease A\n",
        encoding="utf-8",
    )

    summary = prepare_metadata_artifacts(data_root=data_root)

    assert summary is not None
    clean_csv = csv_root / "Medicine_Details_Training.csv"
    vector_csv = csv_root / "Medicine_Details_Training_vectors.csv"
    assert clean_csv.exists()
    assert vector_csv.exists()

