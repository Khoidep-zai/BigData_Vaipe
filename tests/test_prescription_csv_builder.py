import json
from pathlib import Path
from typing import cast

import pandas as pd
from PIL import Image

from src.data.prescription_csv_builder import build_prescription_csv_artifacts


def _touch_rgb_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=(90, 120, 180)).save(path)


def test_build_prescription_csv_artifacts(tmp_path: Path):
    image_root = tmp_path / "data" / "images"
    csv_root = tmp_path / "data" / "csv"

    train_root = image_root / "public_train"
    test_root = image_root / "public_test"

    # Build minimal train split with labels.
    _touch_rgb_image(train_root / "prescription" / "image" / "VAIPE_P_TRAIN_0.jpg")
    _touch_rgb_image(train_root / "pill" / "image" / "VAIPE_P_0_0.jpg")

    (train_root / "prescription" / "label").mkdir(parents=True, exist_ok=True)
    (train_root / "pill" / "label").mkdir(parents=True, exist_ok=True)

    (train_root / "pill_pres_map.json").write_text(
        json.dumps([
            {
                "pres": "VAIPE_P_TRAIN_0.json",
                "pill": ["VAIPE_P_0_0.json"],
            }
        ]),
        encoding="utf-8",
    )

    # Prescription contains class 2 only.
    (train_root / "prescription" / "label" / "VAIPE_P_TRAIN_0.json").write_text(
        json.dumps([
            {"label": "drugname", "mapping": 2},
        ]),
        encoding="utf-8",
    )

    # Pill image has one in-prescription class and one outside class.
    (train_root / "pill" / "label" / "VAIPE_P_0_0.json").write_text(
        json.dumps([
            {"x": 10, "y": 11, "w": 12, "h": 13, "label": 2},
            {"x": 20, "y": 21, "w": 22, "h": 23, "label": 8},
        ]),
        encoding="utf-8",
    )

    # Build minimal test split without labels (real test-like behavior).
    _touch_rgb_image(test_root / "prescription" / "image" / "VAIPE_P_TEST_0.jpg")
    _touch_rgb_image(test_root / "pill" / "image" / "VAIPE_P_100_0.jpg")
    (test_root / "pill_pres_map.json").write_text(
        json.dumps([
            {
                "pres": "VAIPE_P_TEST_0.json",
                "pill": ["VAIPE_P_100_0.json"],
            }
        ]),
        encoding="utf-8",
    )

    # Seed source metadata CSV.
    csv_root.mkdir(parents=True, exist_ok=True)
    source_df = pd.DataFrame(
        [
            {
                "Medicine Name": "Drug class 2",
                "Composition": "Comp 2",
                "VAIPE2022_Class_ID": 2,
                "Dosage_Form": "Tablet",
                "Weight": "",
                "Length_mm": 12.0,
                "Width_mm": 6.0,
                "Height_mm": 3.0,
                "Color_For_AI": "White",
                "Shape_For_AI": "Round",
                "Active_Ingredient_Group": "GroupA",
                "Disease_Treated_VI": "DiseaseA",
                "Data_Source": "unit_test",
            },
            {
                "Medicine Name": "Drug class 8",
                "Composition": "Comp 8",
                "VAIPE2022_Class_ID": 8,
                "Dosage_Form": "Capsule",
                "Weight": "",
                "Length_mm": 13.0,
                "Width_mm": 5.0,
                "Height_mm": 5.0,
                "Color_For_AI": "Blue",
                "Shape_For_AI": "Oval",
                "Active_Ingredient_Group": "GroupB",
                "Disease_Treated_VI": "DiseaseB",
                "Data_Source": "unit_test",
            },
        ]
    )
    source_df.to_csv(csv_root / "Medicine_Details_Deeplearning.csv", index=False, encoding="utf-8-sig")

    summary = build_prescription_csv_artifacts(image_root=image_root, csv_root=csv_root)

    assert summary.prescription_rows == 2
    assert summary.mapping_rows == 2
    assert summary.annotation_rows == 2
    assert summary.metadata_rows == 108
    assert summary.metadata_training_rows == 108
    assert summary.metadata_vector_rows == 108

    ann_df = cast(pd.DataFrame, pd.read_csv(csv_root / "Prescription_Pill_Annotations.csv", encoding="utf-8-sig"))
    assert set(ann_df["target_class_id"].tolist()) == {2, 107}
    assert ann_df.loc[ann_df["true_class_id"] == 2, "is_in_prescription"].iloc[0]
    assert not ann_df.loc[ann_df["true_class_id"] == 8, "is_in_prescription"].iloc[0]

    deep_df = cast(pd.DataFrame, pd.read_csv(csv_root / "Medicine_Details_Deeplearning.csv", encoding="utf-8-sig"))
    assert set(deep_df["VAIPE2022_Class_ID"].astype(int).tolist()) == set(range(108))

    assert (csv_root / "Prescription_Image_Index.csv").exists()
    assert (csv_root / "Prescription_Pill_Map.csv").exists()
    assert (csv_root / "Medicine_Details_Training.csv").exists()
    assert (csv_root / "Medicine_Details_Training_vectors.csv").exists()

