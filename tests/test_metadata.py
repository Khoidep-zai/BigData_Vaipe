from pathlib import Path

from src.data.metadata import MedicineMetadataIndex


def test_metadata_best_match_from_class_name(tmp_path: Path):
    csv_path = tmp_path / "medicine.csv"
    csv_path.write_text(
        "Medicine Name,Composition,Dosage_Form,Weight,Color_For_AI,Shape_For_AI,Active_Ingredient_Group,Disease_Treated_VI\n"
        "Cefadroxil 500mg,Cefadroxil (500mg),Vien nang,14x5x5,Vang,Bau duc,Khang sinh,Nhiem khuan\n"
        "Panactol 500mg,Paracetamol (500mg),Vien nen,10x10x4,Hong,Tron,Giam dau - ha sot,Sot dau nhuc\n",
        encoding="utf-8",
    )

    index = MedicineMetadataIndex.from_csv(csv_path)
    rec = index.best_match("cefadroxil_500mg_0.5g")

    assert rec is not None
    assert "cefadroxil" in rec.medicine_name.lower()


def test_metadata_to_dict_empty_when_none():
    index = MedicineMetadataIndex([])
    assert index.to_dict(None) == {}
