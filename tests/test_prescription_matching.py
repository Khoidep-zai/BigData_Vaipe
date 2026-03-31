from pathlib import Path
from typing import Any, Dict, cast

import pandas as pd

from src.inference.prescription_matching import (
    format_pretty_summary,
    match_pills_to_prescription,
    result_to_dict,
    write_result_csv,
)


def test_match_pills_to_prescription(tmp_path: Path):
    csv_root = tmp_path / "csv"
    csv_root.mkdir(parents=True, exist_ok=True)

    pres_img = "data/images/public_train/prescription/image/VAIPE_P_TRAIN_0.png"
    pill_a = "data/images/public_train/pill/image/VAIPE_P_0_0.jpg"
    pill_b = "data/images/public_train/pill/image/VAIPE_P_0_1.jpg"

    pd.DataFrame(
        [
            {
                "prescription_json": "VAIPE_P_TRAIN_0.json",
                "prescription_image": pres_img,
                "class_ids_in_prescription": "10 47 64",
            }
        ]
    ).to_csv(csv_root / "Prescription_Image_Index.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {
                "prescription_image": pres_img,
                "pill_image": pill_a,
                "bbox_x": 1,
                "bbox_y": 2,
                "bbox_w": 3,
                "bbox_h": 4,
                "target_class_id": 64,
            },
            {
                "prescription_image": pres_img,
                "pill_image": pill_a,
                "bbox_x": 5,
                "bbox_y": 6,
                "bbox_w": 7,
                "bbox_h": 8,
                "target_class_id": 107,
            },
            {
                "prescription_image": pres_img,
                "pill_image": pill_b,
                "bbox_x": 9,
                "bbox_y": 10,
                "bbox_w": 11,
                "bbox_h": 12,
                "target_class_id": 47,
            },
        ]
    ).to_csv(csv_root / "Prescription_Pill_Annotations.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {"VAIPE2022_Class_ID": 47, "Medicine Name": "Drug 47"},
            {"VAIPE2022_Class_ID": 64, "Medicine Name": "Drug 64"},
            {"VAIPE2022_Class_ID": 107, "Medicine Name": "OUT"},
        ]
    ).to_csv(csv_root / "Medicine_Details_Training.csv", index=False, encoding="utf-8-sig")

    result = match_pills_to_prescription(
        prescription_image=pres_img,
        pill_images=[pill_a, pill_b],
        annotations_csv=csv_root / "Prescription_Pill_Annotations.csv",
        prescription_index_csv=csv_root / "Prescription_Image_Index.csv",
        metadata_csv=csv_root / "Medicine_Details_Training.csv",
    )

    payload: Dict[str, Any] = result_to_dict(result)
    assert payload["prescription_json"] == "VAIPE_P_TRAIN_0.json"
    assert payload["classes_in_prescription"] == [10, 47, 64]

    pill0 = payload["pill_results"][0]
    assert len(pill0["in_prescription"]) == 1
    assert len(pill0["out_of_prescription"]) == 1
    assert pill0["in_prescription"][0]["class_id"] == 64
    assert pill0["in_prescription"][0]["medicine_name"] == "Drug 64"
    assert pill0["out_of_prescription"][0]["class_id"] == 107

    pill1 = payload["pill_results"][1]
    assert len(pill1["in_prescription"]) == 1
    assert len(pill1["out_of_prescription"]) == 0
    assert pill1["in_prescription"][0]["class_id"] == 47

    summary_text = format_pretty_summary(result)
    assert "total_in_prescription: 2" in summary_text
    assert "total_out_of_prescription: 1" in summary_text

    csv_path = write_result_csv(result, csv_root / "match_output.csv")
    out_df = cast(pd.DataFrame, pd.read_csv(csv_path, encoding="utf-8-sig"))
    assert len(out_df) == 3
    assert set(out_df["group"].tolist()) == {"in_prescription", "out_of_prescription"}

