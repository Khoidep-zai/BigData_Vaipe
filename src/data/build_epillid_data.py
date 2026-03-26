from __future__ import annotations

"""
Script tích hợp dữ liệu ePillID vào THUOC.

Tính năng chính:
- Đọc params.json và CSV split all/val/test từ ePillID.
- Tạo data split tương thích với THUOC:
    output_root/train/<class_name>/*
    output_root/val/<class_name>/*
    output_root/test/<class_name>/*
- Hỗ trợ lựa chọn nhãn:
    + pilltype: class theo pilltype_id
    + appearance: class theo pilltype_id + side (front/back)
- Hỗ trợ file operation tối ưu tốc độ/đĩa:
    + copy (mặc định)
    + hardlink
    + symlink
- Xuất:
    + pill_metadata.csv
    + class_to_idx.json
    + summary.json
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm


def _resolve_img_path(img_root: Path, img_path: str) -> Path:
    p = Path(str(img_path))
    if p.is_absolute():
        return p
    # Ưu tiên path tương đối theo img_root, fallback theo cwd hiện tại.
    candidate = img_root / p
    if candidate.exists():
        return candidate
    return p


def _load_params(epillid_root: Path) -> Dict[str, object]:
    params_path = epillid_root / "src" / "configs" / "params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {params_path}")
    with params_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_class_name(raw: str) -> str:
    txt = str(raw).strip()
    txt = txt.replace("\\", "_").replace("/", "_")
    txt = txt.replace(" ", "_")
    return txt


def _build_class_name(row: pd.Series, label_mode: str, label_col: str) -> str:
    base = _safe_class_name(row[label_col])
    if label_mode == "pilltype":
        return base

    is_front = row.get("is_front", None)
    side = "unknown"
    if is_front is not None:
        side = "front" if int(is_front) == 1 else "back"
    return f"{base}_{side}"


def _file_op_copy(src_path: Path, dst_path: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(src_path, dst_path)
        return

    if mode == "hardlink":
        try:
            os.link(src_path, dst_path)
            return
        except OSError:
            shutil.copy2(src_path, dst_path)
            return

    if mode == "symlink":
        try:
            os.symlink(src_path, dst_path)
            return
        except OSError:
            shutil.copy2(src_path, dst_path)
            return

    raise ValueError(f"Unsupported file operation mode: {mode}")


def load_epillid_splits(epillid_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc all/val/test CSV theo params.json của ePillID."""
    params = _load_params(epillid_root)

    all_csv = epillid_root / str(params["all_imgs_csv"])
    val_csv = epillid_root / str(params["val_imgs_csv"])
    test_csv = epillid_root / str(params["test_imgs_csv"])

    if not all_csv.exists() or not val_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            f"Không tìm thấy một trong các file CSV: {all_csv}, {val_csv}, {test_csv}. "
            "Hãy đảm bảo bạn đã giải nén đầy đủ ePillID folds/CSV."
        )

    all_df = pd.read_csv(all_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    return all_df, val_df, test_df


def build_thuoc_data_from_epillid(
    epillid_root: Path,
    img_root: Path,
    output_root: Path,
    label_col: str = "pilltype_id",
    label_mode: str = "pilltype",
    file_op_mode: str = "copy",
) -> None:
    """Xây dữ liệu THUOC từ ePillID."""
    all_df, val_df, test_df = load_epillid_splits(epillid_root)

    # Xác định train = all - val - test
    used_paths = pd.concat([val_df["image_path"], test_df["image_path"]])
    train_df = all_df[~all_df["image_path"].isin(used_paths)].copy()

    # Xóa dữ liệu cũ
    if output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "train").mkdir(parents=True, exist_ok=True)
    (output_root / "val").mkdir(parents=True, exist_ok=True)
    (output_root / "test").mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    class_counts: Dict[str, Dict[str, int]] = {}

    def _inc_count(split_name: str, class_name: str) -> None:
        if class_name not in class_counts:
            class_counts[class_name] = {"train": 0, "val": 0, "test": 0}
        class_counts[class_name][split_name] += 1

    def _copy_split(split_name: str, df: pd.DataFrame) -> None:
        nonlocal metadata_rows
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copy {split_name}"):
            cls = _build_class_name(row, label_mode=label_mode, label_col=label_col)
            src_path = _resolve_img_path(img_root, row["image_path"])
            if not src_path.exists():
                # bỏ qua ảnh bị thiếu
                continue

            dst_dir = output_root / split_name / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = dst_dir / src_path.name
            _file_op_copy(src_path, dst_path, file_op_mode)
            _inc_count(split_name, cls)

            metadata_rows.append(
                {
                    "split": split_name,
                    "class_name": cls,
                    "pilltype_id": row.get("pilltype_id", row.get(label_col, None)),
                    "label_raw": row.get(label_col, None),
                    "image_path": str(src_path),
                    "dest_path": str(dst_path),
                    "is_ref": row.get("is_ref", None),
                    "is_front": row.get("is_front", None),
                }
            )

    _copy_split("train", train_df)
    _copy_split("val", val_df)
    _copy_split("test", test_df)

    if metadata_rows:
        meta_df = pd.DataFrame(metadata_rows)
        meta_df.to_csv(output_root / "pill_metadata.csv", index=False)

    # Build class_to_idx sorted by class names for reproducibility.
    classes = sorted(class_counts.keys())
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    with (output_root / "class_to_idx.json").open("w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

    summary = {
        "epillid_root": str(epillid_root),
        "img_root": str(img_root),
        "output_root": str(output_root),
        "label_col": label_col,
        "label_mode": label_mode,
        "file_op_mode": file_op_mode,
        "num_classes": len(classes),
        "num_samples": {
            "train": int(sum(v["train"] for v in class_counts.values())),
            "val": int(sum(v["val"] for v in class_counts.values())),
            "test": int(sum(v["test"] for v in class_counts.values())),
        },
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[DONE] ePillID integration summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tích hợp dữ liệu ePillID vào cấu trúc dữ liệu của THUOC"
    )
    parser.add_argument(
        "--epillid-root",
        type=str,
        required=True,
        help="Thư mục gốc của ePillID-benchmark-ePillID_data_v1.0",
    )
    parser.add_argument(
        "--img-root",
        type=str,
        default=None,
        help="Thư mục gốc chứa ảnh vật lý (mặc định = epillid-root)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data_epillid"),
        help="Thư mục đầu ra chứa train/val/test",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="pilltype_id",
        help="Tên cột nhãn trong CSV split ePillID",
    )
    parser.add_argument(
        "--label-mode",
        type=str,
        default="pilltype",
        choices=["pilltype", "appearance"],
        help="pilltype: class theo pilltype_id, appearance: class theo pilltype_id + front/back",
    )
    parser.add_argument(
        "--file-op",
        type=str,
        default="copy",
        choices=["copy", "hardlink", "symlink"],
        help="Cách đưa ảnh sang output để tối ưu tốc độ/dung lượng",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    epillid_root = Path(args.epillid_root).resolve()
    img_root = Path(args.img_root).resolve() if args.img_root else epillid_root
    output_root = Path(args.output_root).resolve()

    build_thuoc_data_from_epillid(
        epillid_root=epillid_root,
        img_root=img_root,
        output_root=output_root,
        label_col=args.label_col,
        label_mode=args.label_mode,
        file_op_mode=args.file_op,
    )


if __name__ == "__main__":
    main()

