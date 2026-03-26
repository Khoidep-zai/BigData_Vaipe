from __future__ import annotations

import csv
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set


def _strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    """Normalize medicine text to improve robust matching between folder names and CSV fields."""
    # Việc chuẩn hóa này giúp các tên lớp như 'cefadroxil_500mg_0.5g' có thể khớp được với dòng tương ứng trong file CSV.
    txt = _strip_accents((text or "").lower())
    txt = txt.replace("_", " ").replace("-", " ")
    txt = txt.replace(",", ".")
    txt = re.sub(r"[^a-z0-9\.\s]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _tokenize(text: str) -> Set[str]:
    norm = normalize_text(text)
    tokens = {tok for tok in norm.split(" ") if tok}
    return tokens


def _hash_token_to_bucket(token: str, dim: int) -> int:
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % dim


def text_to_hashed_vector(text: str, dim: int = 32) -> List[float]:
    """Chuyen chuoi text thanh vector so co kich thuoc co dinh bang hashing-trick."""
    vec = [0.0] * dim
    for tok in _tokenize(text):
        vec[_hash_token_to_bucket(tok, dim)] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


@dataclass(frozen=True)
class MedicineMetadataRecord:
    medicine_name: str
    composition: str
    dosage_form: str
    weight: str
    color: str
    shape: str
    active_group: str
    disease_vi: str


class MedicineMetadataIndex:
    """In-memory index to match class names (folder labels) with CSV medicine rows."""

    def __init__(self, records: List[MedicineMetadataRecord]) -> None:
        self.records = records
        self._tokens_by_idx: List[Set[str]] = []
        for r in records:
            # Sử dụng kết hợp tên thuốc + thành phần + bệnh điều trị để tăng khả năng tìm kiếm khớp theo ngữ nghĩa rộng hơn.
            token_text = " ".join([r.medicine_name, r.composition, r.disease_vi])
            self._tokens_by_idx.append(_tokenize(token_text))

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "MedicineMetadataIndex":
        path = Path(csv_path)
        if not path.exists():
            return cls([])

        records: List[MedicineMetadataRecord] = []
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(
                    MedicineMetadataRecord(
                        medicine_name=(row.get("Medicine Name") or "").strip(),
                        composition=(row.get("Composition") or "").strip(),
                        dosage_form=(row.get("Dosage_Form") or "").strip(),
                        weight=(row.get("Weight") or "").strip(),
                        color=(row.get("Color_For_AI") or "").strip(),
                        shape=(row.get("Shape_For_AI") or "").strip(),
                        active_group=(row.get("Active_Ingredient_Group") or "").strip(),
                        disease_vi=(row.get("Disease_Treated_VI") or "").strip(),
                    )
                )
        return cls(records)

    def best_match(self, class_name: str) -> Optional[MedicineMetadataRecord]:
        """Find best metadata row for class/folder name by token overlap score."""
        if not self.records:
            return None

        # Tìm dòng thông tin thuốc phù hợp nhất cho tên lớp/thư mục dựa trên điểm số trùng lặp từ khóa.
        query_tokens = _tokenize(class_name)
        if not query_tokens:
            return None

        best_score = 0.0
        best_idx = -1

        for idx, tokens in enumerate(self._tokens_by_idx):
            if not tokens:
                continue
            inter = len(query_tokens & tokens)
            if inter == 0:
                continue

            # So khop kieu Dice: giup ket qua on dinh ke ca khi ten lop ngan hon nhieu so voi mo ta day du trong CSV.
            score = (2.0 * inter) / (len(query_tokens) + len(tokens))
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx < 0:
            return None

        # Prevent weak accidental matches.
        if best_score < 0.2:
            return None

        return self.records[best_idx]

    def to_dict(self, record: Optional[MedicineMetadataRecord]) -> Dict[str, str]:
        if record is None:
            return {}
        return {
            "medicine_name": record.medicine_name,
            "composition": record.composition,
            "dosage_form": record.dosage_form,
            "weight": record.weight,
            "color": record.color,
            "shape": record.shape,
            "active_group": record.active_group,
            "disease_vi": record.disease_vi,
        }

    def to_numeric_vector(
        self,
        record: Optional[MedicineMetadataRecord],
        text_dim: int = 32,
    ) -> Dict[str, float]:
        """
        So hoa metadata thanh vector so de dung cho phan tich/feature fusion.
        Khong thay doi model hien tai, chi cung cap du lieu vector bo sung.
        """
        if record is None:
            return {f"meta_{i:03d}": 0.0 for i in range(text_dim * 3)}

        name_vec = text_to_hashed_vector(record.medicine_name, dim=text_dim)
        comp_vec = text_to_hashed_vector(record.composition, dim=text_dim)
        disease_vec = text_to_hashed_vector(record.disease_vi, dim=text_dim)

        out: Dict[str, float] = {}
        offset = 0
        for chunk in [name_vec, comp_vec, disease_vec]:
            for i, v in enumerate(chunk):
                out[f"meta_{offset + i:03d}"] = float(v)
            offset += text_dim
        return out


def export_metadata_vectors_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    text_dim: int = 32,
) -> None:
    """Doc metadata CSV va xuat ban so hoa (vector) de cac pipeline phan tich co the tai su dung."""
    index = MedicineMetadataIndex.from_csv(input_csv)
    output_csv = Path(output_csv)
    rows: List[Dict[str, object]] = []

    for rec in index.records:
        base = {
            "medicine_name": rec.medicine_name,
            "composition": rec.composition,
            "dosage_form": rec.dosage_form,
            "weight": rec.weight,
            "color": rec.color,
            "shape": rec.shape,
            "active_group": rec.active_group,
            "disease_vi": rec.disease_vi,
        }
        base.update(index.to_numeric_vector(rec, text_dim=text_dim))
        rows.append(base)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["medicine_name"])
        return

    fieldnames = list(rows[0].keys())
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
