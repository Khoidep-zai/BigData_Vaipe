from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, cast

import pandas as pd


def _read_csv_df(path: str | Path) -> pd.DataFrame:
    return cast(pd.DataFrame, pd.read_csv(path, encoding="utf-8-sig"))


@dataclass
class PillDetection:
    class_id: int
    class_name: str
    medicine_name: str
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int


@dataclass
class PillImageMatchResult:
    pill_image: str
    in_prescription: List[PillDetection]
    out_of_prescription: List[PillDetection]


@dataclass
class PrescriptionMatchResult:
    prescription_image: str
    prescription_json: str
    classes_in_prescription: List[int]
    pill_results: List[PillImageMatchResult]


def _normalize_path(value: str) -> str:
    text = (value or "").strip().replace("\\", "/")
    return text.lower()


def _parse_class_ids(value: str) -> List[int]:
    out: List[int] = []
    for token in str(value or "").split():
        try:
            cid = int(token)
        except ValueError:
            continue
        if 0 <= cid <= 106:
            out.append(cid)
    return sorted(set(out))


def _build_medicine_name_map(metadata_csv: str | Path) -> Dict[int, str]:
    path = Path(metadata_csv)
    if not path.exists():
        return {}

    df = _read_csv_df(path)
    if "VAIPE2022_Class_ID" not in df.columns or "Medicine Name" not in df.columns:
        return {}

    mapping: Dict[int, str] = {}
    for _, row in df.iterrows():
        try:
            class_id = int(row["VAIPE2022_Class_ID"])
        except (TypeError, ValueError):
            continue
        if class_id < 0 or class_id > 107:
            continue
        medicine_name = str(row.get("Medicine Name", "")).strip()
        if medicine_name:
            mapping[class_id] = medicine_name

    if 107 not in mapping:
        mapping[107] = "OUT_OF_PRESCRIPTION"
    return mapping


def _load_prescription_context(
    prescription_image: str,
    prescription_index_csv: str | Path,
) -> tuple[str, List[int]]:
    path = Path(prescription_index_csv)
    if not path.exists():
        return "", []

    df = _read_csv_df(path)
    if "prescription_image" not in df.columns:
        return "", []

    target_norm = _normalize_path(prescription_image)
    target_name = Path(target_norm).name

    for _, row in df.iterrows():
        row_img = str(row.get("prescription_image", ""))
        row_norm = _normalize_path(row_img)
        if row_norm == target_norm or Path(row_norm).name == target_name:
            pres_json = str(row.get("prescription_json", "")).strip()
            classes = _parse_class_ids(str(row.get("class_ids_in_prescription", "")))
            return pres_json, classes

    return "", []


def _detections_for_pill_rows(
    pill_rows: pd.DataFrame,
    medicine_name_map: Dict[int, str],
) -> PillImageMatchResult:
    in_items: List[PillDetection] = []
    out_items: List[PillDetection] = []

    if pill_rows.empty:
        return PillImageMatchResult(pill_image="", in_prescription=in_items, out_of_prescription=out_items)

    pill_image = str(pill_rows.iloc[0].get("pill_image", "")).strip()
    for _, row in pill_rows.iterrows():
        target_class_id = int(row.get("target_class_id", 107))
        class_name = f"class_{target_class_id}"
        medicine_name = medicine_name_map.get(target_class_id, class_name)

        detection = PillDetection(
            class_id=target_class_id,
            class_name=class_name,
            medicine_name=medicine_name,
            bbox_x=int(row.get("bbox_x", 0)),
            bbox_y=int(row.get("bbox_y", 0)),
            bbox_w=int(row.get("bbox_w", 0)),
            bbox_h=int(row.get("bbox_h", 0)),
        )
        if target_class_id == 107:
            out_items.append(detection)
        else:
            in_items.append(detection)

    return PillImageMatchResult(
        pill_image=pill_image,
        in_prescription=in_items,
        out_of_prescription=out_items,
    )


def match_pills_to_prescription(
    *,
    prescription_image: str,
    pill_images: Iterable[str],
    annotations_csv: str | Path = "data/csv/Prescription_Pill_Annotations.csv",
    prescription_index_csv: str | Path = "data/csv/Prescription_Image_Index.csv",
    metadata_csv: str | Path = "data/csv/Medicine_Details_Training.csv",
) -> PrescriptionMatchResult:
    """
    Match each pill image against a specific prescription context.

    Output groups detections into:
    - in_prescription: class_0..class_106
    - out_of_prescription: class_107
    """
    ann_path = Path(annotations_csv)
    if not ann_path.exists():
        raise FileNotFoundError(f"Khong tim thay annotation CSV: {ann_path}")

    medicine_name_map = _build_medicine_name_map(metadata_csv)
    pres_json, classes_in_prescription = _load_prescription_context(
        prescription_image,
        prescription_index_csv,
    )

    ann_df = _read_csv_df(ann_path)
    for required_col in [
        "prescription_image",
        "pill_image",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "target_class_id",
    ]:
        if required_col not in ann_df.columns:
            raise RuntimeError(f"Thieu cot bat buoc trong annotation CSV: {required_col}")

    target_pres_norm = _normalize_path(prescription_image)
    target_pres_name = Path(target_pres_norm).name
    ann_df["__pres_norm"] = ann_df["prescription_image"].astype(str).map(_normalize_path)
    ann_df = ann_df[
        (ann_df["__pres_norm"] == target_pres_norm)
        | (ann_df["__pres_norm"].map(lambda x: Path(x).name == target_pres_name))
    ].copy()

    results: List[PillImageMatchResult] = []
    for pill_image in pill_images:
        pill_norm = _normalize_path(pill_image)
        pill_name = Path(pill_norm).name

        pill_rows = ann_df[
            (ann_df["pill_image"].astype(str).map(_normalize_path) == pill_norm)
            | (ann_df["pill_image"].astype(str).map(lambda x: Path(_normalize_path(x)).name == pill_name))
        ].copy()

        if pill_rows.empty:
            results.append(
                PillImageMatchResult(
                    pill_image=str(pill_image),
                    in_prescription=[],
                    out_of_prescription=[],
                )
            )
            continue

        one = _detections_for_pill_rows(pill_rows, medicine_name_map)
        if not one.pill_image:
            one.pill_image = str(pill_image)
        results.append(one)

    return PrescriptionMatchResult(
        prescription_image=str(prescription_image),
        prescription_json=pres_json,
        classes_in_prescription=classes_in_prescription,
        pill_results=results,
    )


def result_to_dict(result: PrescriptionMatchResult) -> Dict[str, Any]:
    return {
        "prescription_image": result.prescription_image,
        "prescription_json": result.prescription_json,
        "classes_in_prescription": result.classes_in_prescription,
        "pill_results": [
            {
                "pill_image": item.pill_image,
                "in_prescription": [asdict(x) for x in item.in_prescription],
                "out_of_prescription": [asdict(x) for x in item.out_of_prescription],
            }
            for item in result.pill_results
        ],
    }


def result_to_rows(result: PrescriptionMatchResult) -> List[Dict[str, Any]]:
    """Flatten match result into one-row-per-detection records for CSV export."""
    rows: List[Dict[str, Any]] = []
    for pill in result.pill_results:
        for det in pill.in_prescription:
            rows.append(
                {
                    "prescription_image": result.prescription_image,
                    "prescription_json": result.prescription_json,
                    "pill_image": pill.pill_image,
                    "group": "in_prescription",
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                    "medicine_name": det.medicine_name,
                    "bbox_x": det.bbox_x,
                    "bbox_y": det.bbox_y,
                    "bbox_w": det.bbox_w,
                    "bbox_h": det.bbox_h,
                }
            )
        for det in pill.out_of_prescription:
            rows.append(
                {
                    "prescription_image": result.prescription_image,
                    "prescription_json": result.prescription_json,
                    "pill_image": pill.pill_image,
                    "group": "out_of_prescription",
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                    "medicine_name": det.medicine_name,
                    "bbox_x": det.bbox_x,
                    "bbox_y": det.bbox_y,
                    "bbox_w": det.bbox_w,
                    "bbox_h": det.bbox_h,
                }
            )
    return rows


def write_result_csv(result: PrescriptionMatchResult, output_csv: str | Path) -> Path:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = result_to_rows(result)

    if rows:
        pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(
            columns=[
                "prescription_image",
                "prescription_json",
                "pill_image",
                "group",
                "class_id",
                "class_name",
                "medicine_name",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
            ]
        ).to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def format_pretty_summary(result: PrescriptionMatchResult) -> str:
    total_in = sum(len(p.in_prescription) for p in result.pill_results)
    total_out = sum(len(p.out_of_prescription) for p in result.pill_results)
    lines = [
        "[SUMMARY] Prescription matching",
        f"- prescription: {result.prescription_image}",
        f"- prescription_json: {result.prescription_json or '(unknown)'}",
        f"- classes_in_prescription: {result.classes_in_prescription}",
        f"- total_pill_images: {len(result.pill_results)}",
        f"- total_in_prescription: {total_in}",
        f"- total_out_of_prescription: {total_out}",
    ]

    for item in result.pill_results:
        lines.append(
            f"  * {item.pill_image}: in={len(item.in_prescription)}, out={len(item.out_of_prescription)}"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Match pill detections to one prescription and split results into in/out prescription."
        )
    )
    parser.add_argument("--prescription-image", required=True, help="Path to one prescription image.")
    parser.add_argument(
        "--pill-images",
        required=True,
        nargs="+",
        help="One or more pill image paths captured for this prescription.",
    )
    parser.add_argument(
        "--annotations-csv",
        default="data/csv/Prescription_Pill_Annotations.csv",
        help="Annotation CSV generated by Loc_du_lieu pipeline.",
    )
    parser.add_argument(
        "--prescription-index-csv",
        default="data/csv/Prescription_Image_Index.csv",
        help="Prescription index CSV generated by Loc_du_lieu pipeline.",
    )
    parser.add_argument(
        "--metadata-csv",
        default="data/csv/Medicine_Details_Training.csv",
        help="Metadata CSV used to map class id to medicine name.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional output JSON file path. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Optional output CSV path (one detection per row).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=False,
        help="Print concise terminal summary (counts per pill image).",
    )

    args = parser.parse_args(argv)
    result = match_pills_to_prescription(
        prescription_image=args.prescription_image,
        pill_images=args.pill_images,
        annotations_csv=args.annotations_csv,
        prescription_index_csv=args.prescription_index_csv,
        metadata_csv=args.metadata_csv,
    )
    payload = result_to_dict(result)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] Da ghi ket qua: {output_path}")
    else:
        # ASCII-safe stdout avoids codec errors on Windows cp1252 terminals.
        print(json.dumps(payload, ensure_ascii=True, indent=2))

    if args.output_csv:
        csv_path = write_result_csv(result, args.output_csv)
        print(f"[OK] Da ghi CSV: {csv_path}")

    if args.pretty:
        print(format_pretty_summary(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

