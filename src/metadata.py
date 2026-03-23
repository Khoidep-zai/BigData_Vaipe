from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set


def _strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    """Normalize medicine text to improve robust matching between folder names and CSV fields."""
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

        query_tokens = _tokenize(class_name)
        if not query_tokens:
            return None

        best_idx = -1
        best_score = 0.0

        for idx, tokens in enumerate(self._tokens_by_idx):
            if not tokens:
                continue
            inter = len(query_tokens & tokens)
            if inter == 0:
                continue

            # Dice-like score, robust when class name has fewer tokens than CSV row.
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
