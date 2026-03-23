from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .inference import ComparisonResult, compare_pill_images, compare_pill_images_auto
from .metadata import MedicineMetadataIndex
from .models import load_checkpoint_class_to_idx


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
DEMO_DIR = ROOT_DIR / "demo_images"
METRICS_SUFFIX = "_epillid_best.metrics.json"
METADATA_CSV = ROOT_DIR / "data" / "Medicine_Details_Deeplearning.csv"

# Đuôi ảnh được hỗ trợ
_IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")


@dataclass
class DemoEntry:
    class_name: str
    sample_image_path: str
    query_image_path: Optional[str] = None
    result: Optional[ComparisonResult] = None


class PillClassifierApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Phân loại thuốc bằng ảnh")
        self.geometry("1180x650")
        self.minsize(1100, 600)

        # Làm giao diện sáng sủa, dễ nhìn hơn
        try:
            self.style = ttk.Style(self)
            # Sử dụng theme hiện đại nếu có
            if "clam" in self.style.theme_names():
                self.style.theme_use("clam")
            self.style.configure("TLabel", padding=3)
            self.style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
            self.style.configure("Treeview", rowheight=26, font=("Segoe UI", 9))
            self.style.configure("TButton", font=("Segoe UI", 9))
        except Exception:
            self.style = None

        # Mặc định dùng efficientnet_b0 cho nhẹ và nhanh hơn resnet50/vit
        self.model_name = tk.StringVar(value="efficientnet_b0")

        # Tự động chọn device phù hợp
        import torch
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_str = tk.StringVar(value=default_device)

        self.sim_threshold = tk.DoubleVar(value=0.7)
        self.color_threshold = tk.DoubleVar(value=0.6)
        self.size_threshold = tk.DoubleVar(value=0.6)
        self.texture_threshold = tk.DoubleVar(value=0.6)

        # Flag chống chạy nhiều lần cùng lúc
        self._running = False
        self._warned_mapping_mismatch = False

        self.demo_entries: List[DemoEntry] = []
        self.class_to_idx: Dict[str, int] = {}
        self.metadata_index = MedicineMetadataIndex.from_csv(METADATA_CSV)
        self._load_demo_entries()
        self._build_class_mapping()

        self._build_ui()

    def _build_class_mapping(self) -> None:
        # Map class names from demo_images folder
        classes = sorted(
            [
                d.name
                for d in DEMO_DIR.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]
        )
        self.class_to_idx = {name: i for i, name in enumerate(classes)}

    def _load_demo_entries(self) -> None:
        DEMO_DIR.mkdir(exist_ok=True)
        for class_dir in sorted(
            [d for d in DEMO_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
        ):
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in _IMG_EXTENSIONS:
                    self.demo_entries.append(
                        DemoEntry(class_name=class_dir.name, sample_image_path=str(img_path))
                    )

    def _build_ui(self) -> None:
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        ttk.Label(control_frame, text="Mô hình:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        model_combo = ttk.Combobox(
            control_frame,
            textvariable=self.model_name,
            values=["resnet50", "efficientnet_b0", "vit_b_16", "auto"],
            width=18,
            state="readonly",
        )
        model_combo.pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Thiết bị:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(20, 0))
        device_combo = ttk.Combobox(
            control_frame,
            textvariable=self.device_str,
            values=["cuda", "cpu"],
            width=8,
            state="readonly",
        )
        device_combo.pack(side=tk.LEFT, padx=5)

        # thresholds (compact)
        thresh_frame = ttk.Frame(control_frame)
        thresh_frame.pack(side=tk.LEFT, padx=(20, 0))
        ttk.Label(thresh_frame, text="Sim:").grid(row=0, column=0)
        ttk.Entry(thresh_frame, textvariable=self.sim_threshold, width=4).grid(row=0, column=1)
        ttk.Label(thresh_frame, text="Màu:").grid(row=0, column=2)
        ttk.Entry(thresh_frame, textvariable=self.color_threshold, width=4).grid(row=0, column=3)
        ttk.Label(thresh_frame, text="Kích thước:").grid(row=0, column=4)
        ttk.Entry(thresh_frame, textvariable=self.size_threshold, width=4).grid(row=0, column=5)
        ttk.Label(thresh_frame, text="Texture:").grid(row=0, column=6)
        ttk.Entry(thresh_frame, textvariable=self.texture_threshold, width=4).grid(row=0, column=7)

        # Buttons group (từ phải sang trái)
        self.run_btn = ttk.Button(
            control_frame,
            text="Chạy phân loại cho dòng được chọn",
            command=self.run_selected_row,
        )
        self.run_btn.pack(side=tk.RIGHT, padx=5)

        self.run_all_btn = ttk.Button(
            control_frame,
            text="Chạy tất cả",
            command=self.run_all,
        )
        self.run_all_btn.pack(side=tk.RIGHT, padx=5)

        select_img_btn = ttk.Button(
            control_frame,
            text="Chọn ảnh kiểm tra cho dòng",
            command=self.select_image_for_selected_row,
        )
        select_img_btn.pack(side=tk.RIGHT, padx=5)

        # Short help text for better UX
        help_label = ttk.Label(
            self,
            text=(
                "Hướng dẫn: Nhấp đúp vào từng dòng để chọn ảnh thuốc cần kiểm tra, "
                "sau đó bấm 'Chạy phân loại cho dòng được chọn' hoặc 'Chạy tất cả'. "
                "Có thể tinh chỉnh các ngưỡng Sim/Màu/Kích thước/Texture ở phía trên."
            ),
            wraplength=1000,
            foreground="#444",
            font=("Segoe UI", 9),
        )
        help_label.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 5))

        # Progress bar (ẩn khi không chạy)
        self.progress_frame = ttk.Frame(self)
        self.progress_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        self.progress_label = ttk.Label(self.progress_frame, text="")
        self.progress_label.pack(side=tk.LEFT)
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode="determinate", length=400)
        self.progress_bar.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)
        self.progress_frame.pack_forget()

        # Table frame
        table_frame = ttk.Frame(self)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        columns = ("class_name", "sample_img", "query_img", "result", "details")
        self.tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=15,
        )
        self.tree.heading("class_name", text="Tên thuốc")
        self.tree.heading("sample_img", text="Hình thuốc mẫu")
        self.tree.heading("query_img", text="Ảnh muốn kiểm tra")
        self.tree.heading("result", text="Kết quả (Đúng/Sai)")
        self.tree.heading("details", text="Chi tiết điểm")

        self.tree.column("class_name", width=150, anchor=tk.CENTER)
        self.tree.column("sample_img", width=200, anchor=tk.CENTER)
        self.tree.column("query_img", width=200, anchor=tk.CENTER)
        self.tree.column("result", width=150, anchor=tk.CENTER)
        self.tree.column("details", width=280, anchor=tk.W)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(
            table_frame, orient=tk.VERTICAL, command=self.tree.yview
        )
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bottom: preview images
        preview_frame = ttk.Frame(self)
        preview_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.sample_preview = ttk.Label(preview_frame, text="Hình mẫu")
        self.sample_preview.pack(side=tk.LEFT, padx=20)

        self.query_preview = ttk.Label(preview_frame, text="Ảnh kiểm tra")
        self.query_preview.pack(side=tk.LEFT, padx=20)

        self._populate_table()

        self.tree.bind("<Double-1>", self.on_double_click_row)
        self.tree.bind("<<TreeviewSelect>>", self.on_select_row)

    def _populate_table(self) -> None:
        for idx, entry in enumerate(self.demo_entries):
            self.tree.insert(
                "",
                tk.END,
                iid=str(idx),
                values=(
                    entry.class_name,
                    os.path.basename(entry.sample_image_path),
                    entry.query_image_path or "Chưa chọn",
                    "",
                    "",
                ),
            )

    def _load_image_for_preview(self, path: str, max_size: int = 200) -> ImageTk.PhotoImage:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_size, max_size))
        return ImageTk.PhotoImage(img)

    def on_select_row(self, event=None) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        idx = int(selected[0])
        entry = self.demo_entries[idx]

        # Update previews
        if os.path.exists(entry.sample_image_path):
            sample_photo = self._load_image_for_preview(entry.sample_image_path)
            self.sample_preview.image = sample_photo
            self.sample_preview.configure(image=sample_photo)

        if entry.query_image_path and os.path.exists(entry.query_image_path):
            query_photo = self._load_image_for_preview(entry.query_image_path)
            self.query_preview.image = query_photo
            self.query_preview.configure(image=query_photo)
        else:
            self.query_preview.configure(text="Ảnh kiểm tra", image="")

    def on_double_click_row(self, event) -> None:
        # Shortcut: double-click để chọn ảnh kiểm tra cho dòng
        selected = self.tree.selection()
        if not selected:
            return
        idx = int(selected[0])

        file_path = filedialog.askopenfilename(
            title="Chọn ảnh thuốc cần kiểm tra",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")],
        )
        if not file_path:
            return

        self.demo_entries[idx].query_image_path = file_path
        self.tree.set(selected[0], "query_img", os.path.basename(file_path))
        self.on_select_row()

    def select_image_for_selected_row(self) -> None:
        """Nút riêng để chọn ảnh kiểm tra cho dòng đang được bôi chọn."""
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo(
                "Thông báo",
                "Vui lòng chọn một dòng trong bảng trước, rồi bấm nút 'Chọn ảnh kiểm tra cho dòng'.",
            )
            return
        # Giả lập hành vi double-click nhưng thông qua nút bấm
        self.on_double_click_row(event=None)

    def _get_checkpoint_for_model(self) -> Optional[str]:
        if self.model_name.get() == "auto":
            # Auto mode tự tìm nhiều checkpoint, không dùng 1 file cụ thể
            return None
        pattern = f"{self.model_name.get()}_epillid_best.pt"
        ckpt_path = MODELS_DIR / pattern
        if ckpt_path.exists():
            return str(ckpt_path)
        return None

    def _warn_if_checkpoint_mapping_mismatch(self, ckpt_path: str) -> None:
        """Cảnh báo khi class mapping trong checkpoint khác với demo classes hiện tại."""
        if self._warned_mapping_mismatch:
            return
        try:
            ckpt_map = load_checkpoint_class_to_idx(ckpt_path)
        except Exception:
            return
        if not ckpt_map:
            return

        ckpt_classes = set(ckpt_map.keys())
        demo_classes = set(self.class_to_idx.keys())
        if ckpt_classes != demo_classes:
            self._warned_mapping_mismatch = True
            self.after(
                0,
                lambda: messagebox.showwarning(
                    "Cảnh báo dữ liệu mô hình",
                    "Checkpoint hiện tại được train với bộ class khác dữ liệu demo hiện tại. "
                    "Để kết quả chuẩn nhất, hãy train lại model bằng dữ liệu mới trong THUOC/data.",
                ),
            )

    def _set_buttons_state(self, state: str) -> None:
        """Enable/disable buttons khi đang chạy để tránh lỗi."""
        self.run_btn.config(state=state)
        self.run_all_btn.config(state=state)

    def _show_progress(self, total: int) -> None:
        self.progress_bar["maximum"] = total
        self.progress_bar["value"] = 0
        self.progress_label.config(text="Đang xử lý...")
        self.progress_frame.pack(fill=tk.X, padx=10, pady=(0, 5))

    def _update_progress(self, current: int, total: int) -> None:
        self.progress_bar["value"] = current
        self.progress_label.config(text=f"Đang xử lý... {current}/{total}")

    def _hide_progress(self) -> None:
        self.progress_frame.pack_forget()

    def run_selected_row(self) -> None:
        if self._running:
            return
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo("Thông báo", "Vui lòng chọn một dòng trong bảng.")
            return
        idx = int(selected[0])
        self._running = True
        self._set_buttons_state("disabled")
        # Chạy inference trong thread riêng để không block UI
        thread = threading.Thread(target=self._run_row_thread, args=(idx, True), daemon=True)
        thread.start()

    def run_all(self) -> None:
        if self._running:
            return
        self._running = True
        self._set_buttons_state("disabled")
        # Chạy toàn bộ trong thread riêng
        thread = threading.Thread(target=self._run_all_thread, daemon=True)
        thread.start()

    def _run_all_thread(self) -> None:
        """Thread: chạy inference cho tất cả các dòng có ảnh kiểm tra."""
        rows_with_query = [i for i, e in enumerate(self.demo_entries) if e.query_image_path]
        total = len(rows_with_query)
        if total == 0:
            self.after(0, self._on_inference_done)
            return

        self.after(0, self._show_progress, total)
        for count, idx in enumerate(rows_with_query, 1):
            self._run_row(idx, show_error=False)
            self.after(0, self._update_progress, count, total)
        self.after(0, self._on_inference_done)

    def _run_row_thread(self, idx: int, show_error: bool) -> None:
        """Thread: chạy inference cho 1 dòng."""
        self._run_row(idx, show_error=show_error)
        self.after(0, self._on_inference_done)

    def _on_inference_done(self) -> None:
        """Callback khi inference xong – bật lại buttons, ẩn progress."""
        self._running = False
        self._set_buttons_state("normal")
        self._hide_progress()

    def _run_row(self, idx: int, show_error: bool = True) -> None:
        entry = self.demo_entries[idx]

        if not entry.query_image_path:
            if show_error:
                self.after(0, lambda: messagebox.showwarning(
                    "Thiếu ảnh kiểm tra",
                    "Vui lòng nhấp đúp vào dòng để chọn ảnh thuốc cần kiểm tra.",
                ))
            return

        try:
            if self.model_name.get() == "auto":
                # Kết hợp ưu điểm của tất cả mô hình đã train
                result = compare_pill_images_auto(
                    checkpoint_dir=str(MODELS_DIR),
                    class_to_idx=self.class_to_idx,
                    sample_image_path=entry.sample_image_path,
                    query_image_path=entry.query_image_path,
                    device_str=self.device_str.get(),
                    expected_class_name=entry.class_name,
                    metadata_index=self.metadata_index,
                )
            else:
                ckpt = self._get_checkpoint_for_model()
                if not ckpt:
                    if show_error:
                        self.after(0, lambda: messagebox.showerror(
                            "Không tìm thấy mô hình",
                            "Vui lòng train mô hình và đặt file *_epillid_best.pt trong thư mục models/",
                        ))
                    return
                self._warn_if_checkpoint_mapping_mismatch(ckpt)

                result = compare_pill_images(
                    model_name=self.model_name.get(),
                    checkpoint_path=ckpt,
                    class_to_idx=self.class_to_idx,
                    sample_image_path=entry.sample_image_path,
                    query_image_path=entry.query_image_path,
                    device_str=self.device_str.get(),
                    expected_class_name=entry.class_name,
                    metadata_index=self.metadata_index,
                )
        except Exception as exc:
            if show_error:
                err_msg = str(exc)
                self.after(0, lambda: messagebox.showerror("Lỗi khi suy luận", err_msg))
            return

        entry.result = result
        result_text = (
            f"{'Đúng (True)' if result.is_true else 'Sai (False)'} "
            f"- dự đoán: {result.predicted_class}"
        )
        detail_text = (
            f"sim={result.similarity_score:.2f}, "
            f"màu={result.color_score:.2f}, "
            f"kích={result.size_score:.2f}, "
            f"texture={result.texture_score:.2f}"
        )
        pred_name = result.details.get("pred_medicine_name", "") if isinstance(result.details, dict) else ""
        if pred_name:
            detail_text += f", thuốc={pred_name}"
        # Cập nhật UI từ main thread
        self.after(0, self._update_row_ui, idx, result_text, detail_text)

        # Log ra console
        print(f"[{entry.class_name}] -> {result_text}")
        print(json.dumps(result.details, indent=2))

    def _update_row_ui(self, idx: int, result_text: str, detail_text: str) -> None:
        """Cập nhật dòng trong bảng (phải gọi từ main thread)."""
        self.tree.set(str(idx), "result", result_text)
        self.tree.set(str(idx), "details", detail_text)
        self.on_select_row()


def main() -> None:
    app = PillClassifierApp()
    app.mainloop()


if __name__ == "__main__":
    main()
