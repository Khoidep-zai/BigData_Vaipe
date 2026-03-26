from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image

from .metadata import export_metadata_vectors_csv


DEFAULT_METADATA_KEEP_COLUMNS: List[str] = [
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
class CropSample:
    image_path: Path
    label_file: Path
    ann_idx: int
    class_id: int
    bbox: Tuple[int, int, int, int]


def _class_dirs(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")}


def _sync_split_class_dirs(root: Path) -> bool:
    """
    Ensure train/val/test have identical class directory sets by creating missing
    directories (without moving/deleting images). Returns True when changed.
    """
    split_roots = [root / "train", root / "val", root / "test"]
    if not all(p.exists() for p in split_roots):
        return False

    union_classes: set[str] = set()
    for split_root in split_roots:
        union_classes.update(_class_dirs(split_root))

    changed = False
    for split_root in split_roots:
        for cls_name in union_classes:
            cls_dir = split_root / cls_name
            if not cls_dir.exists():
                cls_dir.mkdir(parents=True, exist_ok=True)
                changed = True
    return changed


def is_valid_dataset_root(root: Path) -> bool:
    if not all((root / split).exists() for split in ("train", "val", "test")):
        return False

    train_classes = _class_dirs(root / "train")
    val_classes = _class_dirs(root / "val")
    test_classes = _class_dirs(root / "test")

    if not train_classes or not val_classes or not test_classes:
        return False

    return train_classes == val_classes == test_classes


def has_vaipe_training_data(raw_root: Path) -> bool:
    image_dir = raw_root / "public_train" / "pill" / "image"
    label_dir = raw_root / "public_train" / "pill" / "label"
    return image_dir.exists() and label_dir.exists()


def _find_image_for_label(image_dir: Path, stem: str) -> Optional[Path]:
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _to_int(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_bbox(
    x: int,
    y: int,
    w: int,
    h: int,
    img_w: int,
    img_h: int,
    margin_ratio: float,
    min_crop_size: int,
) -> Optional[Tuple[int, int, int, int]]:
    if w <= 0 or h <= 0:
        return None

    expand_w = int(round(w * margin_ratio))
    expand_h = int(round(h * margin_ratio))

    x1 = max(0, x - expand_w)
    y1 = max(0, y - expand_h)
    x2 = min(img_w, x + w + expand_w)
    y2 = min(img_h, y + h + expand_h)

    if (x2 - x1) < int(min_crop_size) or (y2 - y1) < int(min_crop_size):
        return None
    return (x1, y1, x2, y2)


def _split_counts(n: int, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    if n <= 2:
        return n, 0, 0

    n_val = max(1, int(round(n * val_ratio)))
    n_test = max(1, int(round(n * test_ratio)))

    while (n - n_val - n_test) < 1:
        if n_test >= n_val and n_test > 1:
            n_test -= 1
            continue
        if n_val > 1:
            n_val -= 1
            continue
        n_val = 0
        n_test = 0
        break

    return n - n_val - n_test, n_val, n_test


def _collect_vaipe_samples(
    raw_root: Path,
    crop_margin: float,
    min_crop_size: int,
) -> Tuple[Dict[int, List[CropSample]], Dict[str, int]]:
    image_dir = raw_root / "public_train" / "pill" / "image"
    label_dir = raw_root / "public_train" / "pill" / "label"

    by_class: Dict[int, List[CropSample]] = {}
    stats = {
        "label_files": 0,
        "missing_images": 0,
        "invalid_annotations": 0,
        "valid_annotations": 0,
    }

    for label_file in sorted(label_dir.glob("*.json")):
        stats["label_files"] += 1
        image_path = _find_image_for_label(image_dir, label_file.stem)
        if image_path is None:
            stats["missing_images"] += 1
            continue

        try:
            anns = json.loads(label_file.read_text(encoding="utf-8"))
        except Exception:
            stats["invalid_annotations"] += 1
            continue

        if not isinstance(anns, list) or not anns:
            continue

        try:
            with Image.open(image_path) as img:
                img_w, img_h = img.size
        except Exception:
            stats["invalid_annotations"] += 1
            continue

        for ann_idx, ann in enumerate(anns):
            if not isinstance(ann, dict):
                stats["invalid_annotations"] += 1
                continue

            class_id = _to_int(ann.get("label"))
            x = _to_int(ann.get("x"))
            y = _to_int(ann.get("y"))
            w = _to_int(ann.get("w"))
            h = _to_int(ann.get("h"))
            if None in (class_id, x, y, w, h):
                stats["invalid_annotations"] += 1
                continue

            bbox = _safe_bbox(
                x=int(x),
                y=int(y),
                w=int(w),
                h=int(h),
                img_w=int(img_w),
                img_h=int(img_h),
                margin_ratio=float(crop_margin),
                min_crop_size=int(min_crop_size),
            )
            if bbox is None:
                stats["invalid_annotations"] += 1
                continue

            sample = CropSample(
                image_path=image_path,
                label_file=label_file,
                ann_idx=int(ann_idx),
                class_id=int(class_id),
                bbox=bbox,
            )
            by_class.setdefault(int(class_id), []).append(sample)
            stats["valid_annotations"] += 1

    return by_class, stats


def build_thuoc_data_from_vaipe(
    raw_root: str | Path,
    output_root: str | Path,
    *,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_samples_per_class: int = 8,
    crop_margin: float = 0.08,
    min_crop_size: int = 24,
    seed: int = 42,
) -> Dict[str, object]:
    raw_root = Path(raw_root)
    output_root = Path(output_root)

    if not has_vaipe_training_data(raw_root):
        raise FileNotFoundError(
            f"Khong tim thay du lieu VAIPE train hop le tai: {raw_root}"
        )

    samples_by_class, collect_stats = _collect_vaipe_samples(
        raw_root=raw_root,
        crop_margin=float(crop_margin),
        min_crop_size=int(min_crop_size),
    )

    if not samples_by_class:
        raise RuntimeError("Khong thu duoc mau hop le nao tu annotation VAIPE.")

    assignments: Dict[str, List[Tuple[str, CropSample]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    class_distribution: Dict[str, Dict[str, int]] = {}
    skipped_classes: Dict[str, int] = {}

    for class_id, samples in sorted(samples_by_class.items(), key=lambda x: x[0]):
        class_name = f"class_{int(class_id):03d}"
        if len(samples) < int(min_samples_per_class):
            skipped_classes[class_name] = len(samples)
            continue

        shuffled = list(samples)
        random.Random(int(seed) + int(class_id)).shuffle(shuffled)

        n_train, n_val, n_test = _split_counts(
            n=len(shuffled),
            val_ratio=float(val_ratio),
            test_ratio=float(test_ratio),
        )

        if n_val == 0 or n_test == 0:
            skipped_classes[class_name] = len(samples)
            continue

        train_samples = shuffled[:n_train]
        val_samples = shuffled[n_train:n_train + n_val]
        test_samples = shuffled[n_train + n_val:n_train + n_val + n_test]

        assignments["train"].extend((class_name, s) for s in train_samples)
        assignments["val"].extend((class_name, s) for s in val_samples)
        assignments["test"].extend((class_name, s) for s in test_samples)

        class_distribution[class_name] = {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        }

    if not class_distribution:
        raise RuntimeError(
            "Khong con class nao du so luong sau khi loc. "
            "Hay giam --min-samples-per-class de thu lai."
        )

    if output_root.exists():
        shutil.rmtree(output_root)
    for split in ("train", "val", "test"):
        (output_root / split).mkdir(parents=True, exist_ok=True)

    metadata_rows: List[Dict[str, object]] = []
    grouped_tasks: Dict[Path, List[Tuple[str, str, CropSample]]] = defaultdict(list)
    for split, items in assignments.items():
        for class_name, sample in items:
            grouped_tasks[sample.image_path].append((split, class_name, sample))

    for image_path, tasks in grouped_tasks.items():
        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")
            for split, class_name, sample in tasks:
                dst_dir = output_root / split / class_name
                dst_dir.mkdir(parents=True, exist_ok=True)
                out_name = f"{sample.image_path.stem}_ann{sample.ann_idx:02d}.jpg"
                dst_path = dst_dir / out_name

                crop = img_rgb.crop(sample.bbox)
                crop.save(dst_path, format="JPEG", quality=95)

                metadata_rows.append(
                    {
                        "split": split,
                        "class_name": class_name,
                        "class_id": int(sample.class_id),
                        "source_image": str(sample.image_path),
                        "source_label_json": str(sample.label_file),
                        "bbox_x1": int(sample.bbox[0]),
                        "bbox_y1": int(sample.bbox[1]),
                        "bbox_x2": int(sample.bbox[2]),
                        "bbox_y2": int(sample.bbox[3]),
                        "dest_image": str(dst_path),
                    }
                )

    classes = sorted(class_distribution.keys())
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    (output_root / "class_to_idx.json").write_text(
        json.dumps(class_to_idx, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(metadata_rows).to_csv(
        output_root / "pill_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary: Dict[str, object] = {
        "source": "VAIPE public_train pill detection labels",
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "num_classes": len(classes),
        "num_samples": {
            split: int(sum(v[split] for v in class_distribution.values()))
            for split in ("train", "val", "test")
        },
        "class_distribution": class_distribution,
        "skipped_classes": skipped_classes,
        "collect_stats": collect_stats,
        "params": {
            "val_ratio": float(val_ratio),
            "test_ratio": float(test_ratio),
            "min_samples_per_class": int(min_samples_per_class),
            "crop_margin": float(crop_margin),
            "min_crop_size": int(min_crop_size),
            "seed": int(seed),
        },
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return summary


def clean_metadata_csv_for_vectors(
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    keep_columns: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Khong tim thay metadata CSV: {input_csv}")

    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    keep = list(keep_columns or DEFAULT_METADATA_KEEP_COLUMNS)
    selected = [c for c in keep if c in df.columns]
    if not selected:
        raise RuntimeError("Khong tim thay cot metadata hop le de giu lai.")

    out = df[selected].copy()

    text_cols = [
        c
        for c in [
            "Medicine Name",
            "Composition",
            "Dosage_Form",
            "Weight",
            "Color_For_AI",
            "Shape_For_AI",
            "Active_Ingredient_Group",
            "Disease_Treated_VI",
            "VAIPE2022_Class_ID",
        ]
        if c in out.columns
    ]
    for col in text_cols:
        out[col] = out[col].fillna("").astype(str).str.strip()

    for col in ["Length_mm", "Width_mm", "Height_mm"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    has_name = (out["Medicine Name"] != "") if "Medicine Name" in out.columns else pd.Series([True] * len(out))
    has_comp = (out["Composition"] != "") if "Composition" in out.columns else pd.Series([True] * len(out))
    out = out[has_name | has_comp].copy()

    dedup_keys = [c for c in ["Medicine Name", "Composition", "VAIPE2022_Class_ID"] if c in out.columns]
    if dedup_keys:
        out = out.drop_duplicates(subset=dedup_keys, keep="first")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    return {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "rows_in": int(len(df)),
        "rows_out": int(len(out)),
        "columns_out": selected,
    }


def prepare_metadata_artifacts(
    data_root: str | Path = "data",
    *,
    raw_csv_name: str = "Medicine_Details_Deeplearning.csv",
    clean_csv_name: str = "Medicine_Details_Training.csv",
    vector_csv_name: str = "Medicine_Details_Training_vectors.csv",
    text_dim: int = 32,
) -> Optional[Dict[str, object]]:
    data_root = Path(data_root)
    input_csv = data_root / raw_csv_name
    if not input_csv.exists():
        return None

    clean_csv = data_root / clean_csv_name
    vector_csv = data_root / vector_csv_name

    # Skip rebuild when artifacts are newer than raw metadata to speed up repeated training runs.
    if clean_csv.exists() and vector_csv.exists():
        raw_mtime = input_csv.stat().st_mtime
        if clean_csv.stat().st_mtime >= raw_mtime and vector_csv.stat().st_mtime >= raw_mtime:
            return {
                "clean_summary": {
                    "input_csv": str(input_csv),
                    "output_csv": str(clean_csv),
                    "skipped": True,
                    "reason": "up_to_date",
                },
                "vector_csv": str(vector_csv),
                "text_dim": int(text_dim),
            }

    clean_summary = clean_metadata_csv_for_vectors(input_csv=input_csv, output_csv=clean_csv)
    export_metadata_vectors_csv(clean_csv, vector_csv, text_dim=int(text_dim))

    return {
        "clean_summary": clean_summary,
        "vector_csv": str(vector_csv),
        "text_dim": int(text_dim),
    }


def discover_or_prepare_data_dir(
    preferred: Optional[str] = None,
    *,
    raw_data_root: str | Path = "data",
    output_root: str | Path = "data_aligned",
    seed: int = 42,
) -> str:
    output_root = Path(output_root)
    raw_data_root = Path(raw_data_root)

    if preferred:
        preferred_path = Path(preferred)
        if is_valid_dataset_root(preferred_path):
            return str(preferred_path)
        if _sync_split_class_dirs(preferred_path) and is_valid_dataset_root(preferred_path):
            print(f"[DATA] Da dong bo class folders train/val/test tai: {preferred_path}")
            return str(preferred_path)

    if is_valid_dataset_root(output_root):
        return str(output_root)
    if _sync_split_class_dirs(output_root) and is_valid_dataset_root(output_root):
        print(f"[DATA] Da dong bo class folders train/val/test tai: {output_root}")
        return str(output_root)

    if preferred:
        preferred_path = Path(preferred)
        if has_vaipe_training_data(preferred_path):
            summary = build_thuoc_data_from_vaipe(
                raw_root=preferred_path,
                output_root=output_root,
                seed=int(seed),
            )
            print(
                "[DATA] Da tao data_aligned tu bo du lieu VAIPE moi: "
                f"{summary.get('num_classes', 0)} classes"
            )
            return str(output_root)

    if has_vaipe_training_data(raw_data_root):
        summary = build_thuoc_data_from_vaipe(
            raw_root=raw_data_root,
            output_root=output_root,
            seed=int(seed),
        )
        print(
            "[DATA] Da tao data_aligned tu bo du lieu VAIPE moi: "
            f"{summary.get('num_classes', 0)} classes"
        )
        return str(output_root)

    raise FileNotFoundError(
        "Khong tim thay dataset train/val/test hop le va cung khong tim thay "
        "du lieu VAIPE raw de tu dong tao data_aligned."
    )
