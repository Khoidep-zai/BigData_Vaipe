from pathlib import Path

from src.data.metadata import (
    MedicineMetadataIndex,
    export_metadata_vectors_csv,
    text_to_hashed_vector,
)


def test_metadata_best_match_from_class_name(tmp_path: Path):
    # Đảm bảo rằng việc tách từ khoá từ tên lớp (tên thư mục) có thể giúp tìm đúng dòng dữ liệu thuốc tương ứng trong file CSV.
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
    # Kiểm tra phòng thủ: nếu bản ghi là None thì phải trả về dict rỗng chứ không gây lỗi.
    index = MedicineMetadataIndex([])
    assert index.to_dict(None) == {}


def test_text_to_hashed_vector_fixed_dim():
    vec = text_to_hashed_vector("cefadroxil 500mg", dim=16)
    assert len(vec) == 16
    assert abs(sum(v * v for v in vec) - 1.0) < 1e-6


def test_export_metadata_vectors_csv(tmp_path: Path):
    csv_path = tmp_path / "medicine.csv"
    out_path = tmp_path / "medicine_vectors.csv"
    csv_path.write_text(
        "Medicine Name,Composition,Dosage_Form,Weight,Color_For_AI,Shape_For_AI,Active_Ingredient_Group,Disease_Treated_VI\n"
        "Cefadroxil 500mg,Cefadroxil (500mg),Vien nang,14x5x5,Vang,Bau duc,Khang sinh,Nhiem khuan\n",
        encoding="utf-8",
    )

    export_metadata_vectors_csv(csv_path, out_path, text_dim=8)
    content = out_path.read_text(encoding="utf-8")
    assert "meta_000" in content
    assert "medicine_name" in content
