from __future__ import annotations

"""
Chức năng hỗ trợ "tự học" từ các kết quả sai.

Ý tưởng ở mức hệ thống (phù hợp đồ án):
- Khi người dùng kiểm tra một cặp ảnh (mẫu, ảnh kiểm tra) và thấy kết quả Sai/Đúng,
  ta có thể lưu lại phản hồi này (feedback) dưới dạng bản ghi.
- Sau đó dùng các bản ghi này để:
  + Tạo một tập dữ liệu "khó" (hard examples) để fine-tune lại mô hình.
  + Hoặc dùng để phân tích thống kê xem mô hình hay nhầm ở lớp nào / điều kiện nào.

File này cung cấp:
- log_feedback: lưu 1 bản ghi feedback (JSON lines).
- load_feedback: đọc lại toàn bộ feedback đã lưu.
- build_hard_example_lists: tách danh sách mẫu sai để dựng lại dataset fine-tune.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List


DEFAULT_FEEDBACK_PATH = Path(__file__).resolve().parents[1] / "data" / "feedback_logs.jsonl"


@dataclass
class FeedbackRecord:
    # Cấu trúc bản ghi tối giản mô tả một lần người dùng kiểm tra kết quả.
    sample_image_path: str  # ảnh mẫu ban đầu
    query_image_path: str  # ảnh người dùng muốn kiểm tra
    predicted_class: str
    is_true_system: bool  # hệ thống kết luận Đúng/Sai
    is_true_user: bool | None  # người dùng đánh giá lại (True nếu thực sự đúng, False nếu sai, None nếu chưa đánh dấu)
    model_name: str  # resnet50 / efficientnet_b0 / vit_b_16 / auto
    similarity_score: float
    color_score: float
    size_score: float
    texture_score: float


def log_feedback(
    record: FeedbackRecord,
    log_path: Path | None = None,
) -> None:
    """
    Ghi 1 bản ghi feedback ra file JSONL.

    - Mỗi dòng là một JSON object (FeedbackRecord).
    - Không chặn luồng chính của GUI (ghi rất nhẹ).
    """
    path = log_path or DEFAULT_FEEDBACK_PATH
    # Dùng định dạng JSONL (mỗi dòng 1 JSON) giúp việc ghi thêm dữ liệu dễ dàng và ít lỗi hơn so với JSON thường.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        json.dump(asdict(record), f, ensure_ascii=False)
        f.write("\n")


def load_feedback(log_path: Path | None = None) -> List[FeedbackRecord]:
    """Đọc toàn bộ feedback đã lưu thành danh sách FeedbackRecord."""
    path = log_path or DEFAULT_FEEDBACK_PATH
    if not path.exists():
        return []

    records: List[FeedbackRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(FeedbackRecord(**data))
            except Exception:
                # Skip malformed lines so one bad row does not break full feedback loading.
                continue
    return records


def build_hard_example_lists(
    records: Iterable[FeedbackRecord],
) -> dict:
    """
    Tạo ra 2 danh sách:
    - system_wrong: hệ thống nói Đúng nhưng người dùng bảo Sai, hoặc ngược lại.
    - hard_cases: các mẫu mà similarity cao nhưng hệ thống đánh giá Sai (gần ranh giới quyết định).

    Kết quả trả về là dict có thể dùng để:
    - Sinh lại danh sách file ảnh để fine-tune mô hình.
    - Phân tích thêm trong notebook/báo cáo.
    """
    system_wrong: list[tuple[str, str]] = []
    hard_cases: list[tuple[str, str]] = []

    for r in records:
        if r.is_true_user is None:
            # Chưa có đánh giá người dùng, khó kết luận
            continue

        # Hệ thống sai so với người dùng
        if r.is_true_system != r.is_true_user:
            system_wrong.append((r.sample_image_path, r.query_image_path))

        # Mẫu "khó": similarity cao nhưng hệ thống đánh giá Sai
        if (not r.is_true_system) and r.similarity_score >= 0.7:
            hard_cases.append((r.sample_image_path, r.query_image_path))

    return {
        "system_wrong": system_wrong,
        "hard_cases": hard_cases,
    }

