from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from .metadata import export_metadata_vectors_csv


METADATA_KEEP_COLUMNS: List[str] = [
    "Medicine Name",
    "Composition",
    "VAIPE2022_Class_ID",
    "Dosage_Form",
    "Weight",
    "Length_mm",
    "Width_mm",
    "Height_mm",
    "Color_For_AI",
    "Shape_For_AI",
    "Active_Ingredient_Group",
    "Disease_Treated_VI",
]


@dataclass(frozen=True)
class BuildSummary:
    prescription_rows: int
    mapping_rows: int
    annotation_rows: int
    metadata_rows: int
    metadata_training_rows: int
    metadata_vector_rows: int


def _load_json_list(path: Path) -> List[dict]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _find_image_for_json(image_dir: Path, json_name: str) -> Optional[Path]:
    stem = Path(json_name).stem
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _extract_prescription_classes(prescription_label_file: Path) -> Set[int]:
    classes: Set[int] = set()
    entries = _load_json_list(prescription_label_file)
    for row in entries:
        if str(row.get("label", "")).strip().lower() != "drugname":
            continue
        mapping = row.get("mapping")
        try:
            class_id = int(mapping)
        except (TypeError, ValueError):
            continue
        if 0 <= class_id <= 106:
            classes.add(class_id)
    return classes


def _collect_mapping_rows(
    *,
    image_root: Path,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    prescription_rows: List[Dict[str, object]] = []
    mapping_rows: List[Dict[str, object]] = []
    annotation_rows: List[Dict[str, object]] = []

    for split in ("public_train", "public_test"):
        split_root = image_root / split
        map_file = split_root / "pill_pres_map.json"
        pairs = _load_json_list(map_file)

        pill_image_dir = split_root / "pill" / "image"
        pill_label_dir = split_root / "pill" / "label"
        pres_image_dir = split_root / "prescription" / "image"
        pres_label_dir = split_root / "prescription" / "label"

        for pair in pairs:
            pres_json = str(pair.get("pres", "")).strip()
            if not pres_json:
                continue

            pres_image_path = _find_image_for_json(pres_image_dir, pres_json)
            pres_label_path = pres_label_dir / pres_json
            classes_in_prescription = _extract_prescription_classes(pres_label_path)

            pill_json_list: Sequence[str] = [
                str(name).strip()
                for name in pair.get("pill", [])
                if str(name).strip()
            ]

            prescription_rows.append(
                {
                    "split": split,
                    "prescription_json": pres_json,
                    "prescription_image": str(pres_image_path) if pres_image_path else "",
                    "prescription_label": str(pres_label_path) if pres_label_path.exists() else "",
                    "num_linked_pill_images": int(len(pill_json_list)),
                    "has_prescription_label": bool(pres_label_path.exists()),
                    "class_ids_in_prescription": " ".join(str(c) for c in sorted(classes_in_prescription)),
                }
            )

            for pill_json in pill_json_list:
                pill_image_path = _find_image_for_json(pill_image_dir, pill_json)
                pill_label_path = pill_label_dir / pill_json

                mapping_rows.append(
                    {
                        "split": split,
                        "prescription_json": pres_json,
                        "prescription_image": str(pres_image_path) if pres_image_path else "",
                        "pill_json": pill_json,
                        "pill_image": str(pill_image_path) if pill_image_path else "",
                        "pill_label": str(pill_label_path) if pill_label_path.exists() else "",
                        "has_pill_label": bool(pill_label_path.exists()),
                        "class_ids_in_prescription": " ".join(str(c) for c in sorted(classes_in_prescription)),
                    }
                )

                if not pill_label_path.exists():
                    continue

                for ann_idx, ann in enumerate(_load_json_list(pill_label_path)):
                    label = ann.get("label")
                    x = ann.get("x")
                    y = ann.get("y")
                    w = ann.get("w")
                    h = ann.get("h")
                    try:
                        true_class_id = int(label)
                        x = int(float(x))
                        y = int(float(y))
                        w = int(float(w))
                        h = int(float(h))
                    except (TypeError, ValueError):
                        continue

                    # Outside-prescription class is always remapped to 107.
                    target_class_id = (
                        true_class_id
                        if (0 <= true_class_id <= 106 and true_class_id in classes_in_prescription)
                        else 107
                    )

                    annotation_rows.append(
                        {
                            "split": split,
                            "prescription_json": pres_json,
                            "prescription_image": str(pres_image_path) if pres_image_path else "",
                            "pill_json": pill_json,
                            "pill_image": str(pill_image_path) if pill_image_path else "",
                            "pill_annotation_index": int(ann_idx),
                            "bbox_x": x,
                            "bbox_y": y,
                            "bbox_w": w,
                            "bbox_h": h,
                            "bbox_center_x": float(x + (w / 2.0)),
                            "bbox_center_y": float(y + (h / 2.0)),
                            "true_class_id": int(true_class_id),
                            "target_class_id": int(target_class_id),
                            "target_class_name": f"class_{int(target_class_id)}",
                            "is_in_prescription": bool(target_class_id != 107),
                            "class_ids_in_prescription": " ".join(str(c) for c in sorted(classes_in_prescription)),
                        }
                    )

    return prescription_rows, mapping_rows, annotation_rows


def _backup_csvs(csv_root: Path, file_names: Iterable[str]) -> Path:
    backup_dir = csv_root / "backup" / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in file_names:
        source = csv_root / name
        if source.exists():
            shutil.copy2(source, backup_dir / name)
    return backup_dir


def _build_filtered_metadata(deep_df: pd.DataFrame) -> pd.DataFrame:
    if "VAIPE2022_Class_ID" not in deep_df.columns:
        raise RuntimeError("Missing required column: VAIPE2022_Class_ID")

    working = deep_df.copy()
    working["VAIPE2022_Class_ID"] = pd.to_numeric(
        working["VAIPE2022_Class_ID"],
        errors="coerce",
    )

    in_pres_df = working[working["VAIPE2022_Class_ID"].between(0, 106, inclusive="both")].copy()
    in_pres_df["VAIPE2022_Class_ID"] = in_pres_df["VAIPE2022_Class_ID"].astype(int)
    in_pres_df = in_pres_df.sort_values(["VAIPE2022_Class_ID"]).drop_duplicates(
        subset=["VAIPE2022_Class_ID"],
        keep="first",
    )

    rows: List[Dict[str, object]] = []
    by_class: Dict[int, Dict[str, object]] = {
        int(row["VAIPE2022_Class_ID"]): row.to_dict() for _, row in in_pres_df.iterrows()
    }

    template: Dict[str, object] = {col: "" for col in working.columns}
    if "Data_Source" in template:
        template["Data_Source"] = "generated_prescription_filter"

    for class_id in range(0, 107):
        if class_id in by_class:
            row = dict(by_class[class_id])
        else:
            row = dict(template)
            row["Medicine Name"] = f"class_{class_id}"
            row["Composition"] = ""
            row["VAIPE2022_Class_ID"] = class_id
        row["Class_Name"] = f"class_{class_id}"
        row["Prescription_Role"] = "in_prescription"
        rows.append(row)

    out_row = dict(template)
    out_row["Medicine Name"] = "OUT_OF_PRESCRIPTION"
    out_row["Composition"] = "Any pill not listed in current prescription"
    out_row["VAIPE2022_Class_ID"] = 107
    out_row["Class_Name"] = "class_107"
    out_row["Prescription_Role"] = "out_of_prescription"
    rows.append(out_row)

    out_df = pd.DataFrame(rows)
    if "VAIPE2022_Class_ID" in out_df.columns:
        out_df["VAIPE2022_Class_ID"] = pd.to_numeric(
            out_df["VAIPE2022_Class_ID"],
            errors="coerce",
        ).astype("Int64")
    return out_df


def build_prescription_csv_artifacts(
    *,
    image_root: str | Path = "data/images",
    csv_root: str | Path = "data/csv",
    text_dim: int = 32,
) -> BuildSummary:
    image_root = Path(image_root)
    csv_root = Path(csv_root)
    csv_root.mkdir(parents=True, exist_ok=True)

    source_deep_csv = csv_root / "Medicine_Details_Deeplearning.csv"
    if not source_deep_csv.exists():
        raise FileNotFoundError(f"Khong tim thay metadata CSV: {source_deep_csv}")

    # Keep a recoverable copy before overwriting existing CSV artifacts.
    _backup_csvs(
        csv_root,
        [
            "Medicine_Details_Deeplearning.csv",
            "Medicine_Details_Training.csv",
            "Medicine_Details_Training_vectors.csv",
        ],
    )

    prescription_rows, mapping_rows, annotation_rows = _collect_mapping_rows(image_root=image_root)

    prescription_df = pd.DataFrame(prescription_rows)
    mapping_df = pd.DataFrame(mapping_rows)
    annotation_df = pd.DataFrame(annotation_rows)

    prescription_df.to_csv(
        csv_root / "Prescription_Image_Index.csv",
        index=False,
        encoding="utf-8-sig",
    )
    mapping_df.to_csv(
        csv_root / "Prescription_Pill_Map.csv",
        index=False,
        encoding="utf-8-sig",
    )
    annotation_df.to_csv(
        csv_root / "Prescription_Pill_Annotations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    deep_df = pd.read_csv(source_deep_csv, encoding="utf-8-sig")
    deep_filtered_df = _build_filtered_metadata(deep_df)
    deep_filtered_df.to_csv(source_deep_csv, index=False, encoding="utf-8-sig")

    training_df = deep_filtered_df.copy()
    for col in METADATA_KEEP_COLUMNS:
        if col not in training_df.columns:
            training_df[col] = ""
    training_df = training_df[METADATA_KEEP_COLUMNS]

    text_cols = [
        col
        for col in [
            "Medicine Name",
            "Composition",
            "Dosage_Form",
            "Weight",
            "Color_For_AI",
            "Shape_For_AI",
            "Active_Ingredient_Group",
            "Disease_Treated_VI",
        ]
        if col in training_df.columns
    ]
    for col in text_cols:
        training_df[col] = training_df[col].fillna("").astype(str).str.strip()

    for col in ["Length_mm", "Width_mm", "Height_mm"]:
        if col in training_df.columns:
            training_df[col] = pd.to_numeric(training_df[col], errors="coerce")

    training_csv = csv_root / "Medicine_Details_Training.csv"
    training_df.to_csv(training_csv, index=False, encoding="utf-8-sig")

    vectors_csv = csv_root / "Medicine_Details_Training_vectors.csv"
    export_metadata_vectors_csv(training_csv, vectors_csv, text_dim=int(text_dim))
    vector_df = pd.read_csv(vectors_csv, encoding="utf-8-sig")

    return BuildSummary(
        prescription_rows=int(len(prescription_df)),
        mapping_rows=int(len(mapping_df)),
        annotation_rows=int(len(annotation_df)),
        metadata_rows=int(len(deep_filtered_df)),
        metadata_training_rows=int(len(training_df)),
        metadata_vector_rows=int(len(vector_df)),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Filter prescription data (public_train/public_test), remap outside pills to class_107, "
            "and refresh CSV artifacts in data/csv."
        )
    )
    parser.add_argument("--image-root", default="data/images", help="Root of image data.")
    parser.add_argument("--csv-root", default="data/csv", help="Folder that stores CSV artifacts.")
    parser.add_argument("--text-dim", type=int, default=32, help="Metadata text hashing dimension.")

    args = parser.parse_args(argv)
    summary = build_prescription_csv_artifacts(
        image_root=args.image_root,
        csv_root=args.csv_root,
        text_dim=int(args.text_dim),
    )

    print("[OK] Da cap nhat xong du lieu CSV cho bai toan trong/ngoai don.")
    print(f"  - Prescription rows         : {summary.prescription_rows}")
    print(f"  - Prescription-pill mappings: {summary.mapping_rows}")
    print(f"  - Pill annotations          : {summary.annotation_rows}")
    print(f"  - Metadata rows             : {summary.metadata_rows}")
    print(f"  - Training metadata rows    : {summary.metadata_training_rows}")
    print(f"  - Vector metadata rows      : {summary.metadata_vector_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

