from __future__ import annotations

import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from flask import Flask, jsonify, render_template, request
import torch
import torch.nn.functional as F
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

# Ensure project root is importable when running from Web/backend/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features import build_transforms, focus_on_object, pil_loader
from src.inference.prescription_matching import match_pills_to_prescription, result_to_dict
from src.models.model_factory import load_checkpoint, load_checkpoint_class_to_idx
from src.utils.model_paths import resolve_model_checkpoint_path

WEB_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_ROOT = WEB_ROOT / "frontend"
UPLOAD_ROOT = WEB_ROOT / "uploads"
UPLOAD_PRESCRIPTIONS = UPLOAD_ROOT / "prescriptions"
UPLOAD_PILLS = UPLOAD_ROOT / "pills"

DATA_ALIGNED_DIR = PROJECT_ROOT / "data_aligned"
MODELS_DIR = PROJECT_ROOT / "models"
EVAL_SUMMARY_CSV = MODELS_DIR / "results" / "evaluation" / "evaluation_summary.csv"
TRAIN_TABLE_CSV = MODELS_DIR / "results" / "training" / "training_results_table.csv"
PRESCRIPTION_ANN_CSV = PROJECT_ROOT / "data" / "csv" / "Prescription_Pill_Annotations.csv"
PRESCRIPTION_INDEX_CSV = PROJECT_ROOT / "data" / "csv" / "Prescription_Image_Index.csv"
MEDICINE_METADATA_CSV = PROJECT_ROOT / "data" / "csv" / "Medicine_Details_Training.csv"

MODEL_NAMES: Tuple[str, ...] = ("resnet50", "efficientnet_b0", "vit_b_16")
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_PILL_IMAGES = max(1, int(os.getenv("THUOC_WEB_MAX_PILL_IMAGES", "30")))
MAX_UPLOAD_FILES_PER_DIR = max(30, int(os.getenv("THUOC_WEB_MAX_UPLOAD_FILES", "500")))
OVERVIEW_CACHE_TTL_SEC = max(0, int(os.getenv("THUOC_WEB_OVERVIEW_TTL_SEC", "25")))

_MODEL_CACHE: Dict[Tuple[str, str, str], Tuple[torch.nn.Module, Dict[int, str], Path]] = {}
_TRANSFORM = build_transforms(train=False)
_MEDICINE_NAME_MAP_CACHE: Optional[Dict[int, str]] = None
_PRESCRIPTION_INDEX_CACHE: Optional[pd.DataFrame] = None
_OVERVIEW_CACHE_DATA: Optional[Dict[str, Any]] = None
_OVERVIEW_CACHE_EXPIRES_AT: float = 0.0


def _resolve_device() -> torch.device:
    requested = os.getenv("THUOC_WEB_DEVICE", "cpu").strip().lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = _resolve_device()


def _normalize_path(value: str) -> str:
    return str(value or "").strip().replace("\\", "/").lower()


def _parse_class_id(class_name: str) -> Optional[int]:
    m = re.search(r"(\d+)$", str(class_name or ""))
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _parse_prescription_class_ids(value: Any) -> List[int]:
    out: List[int] = []
    for token in str(value or "").split():
        try:
            cid = int(token)
        except ValueError:
            continue
        if 0 <= cid <= 106:
            out.append(cid)
    return sorted(set(out))


def _allowed_file(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTS


def _trim_upload_dir(target_dir: Path) -> None:
    files = [f for f in target_dir.iterdir() if f.is_file()]
    overflow = len(files) - MAX_UPLOAD_FILES_PER_DIR
    if overflow <= 0:
        return

    files.sort(key=lambda p: p.stat().st_mtime)
    for old_file in files[:overflow]:
        try:
            old_file.unlink(missing_ok=True)
        except OSError:
            continue


def _save_uploaded_file(file_storage, target_dir: Path) -> Tuple[Path, str, str]:
    if not file_storage or not str(file_storage.filename or "").strip():
        raise ValueError("File upload is empty.")

    original_name = Path(str(file_storage.filename).strip()).name
    safe_name = secure_filename(original_name)
    if not safe_name:
        safe_name = "upload.png"

    ext = (Path(original_name).suffix or Path(safe_name).suffix).lower()
    if ext not in ALLOWED_EXTS:
        raise ValueError(f"Unsupported file type: {ext}")

    target_dir.mkdir(parents=True, exist_ok=True)
    _trim_upload_dir(target_dir)

    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    save_path = target_dir / unique_name
    file_storage.save(save_path)
    return save_path, original_name, unique_name


def _parse_top_k(value: Any, default: int = 3, max_k: int = 10) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return max(1, min(parsed, max_k))


def _is_truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


def _load_medicine_name_map() -> Dict[int, str]:
    global _MEDICINE_NAME_MAP_CACHE
    if _MEDICINE_NAME_MAP_CACHE is not None:
        return _MEDICINE_NAME_MAP_CACHE

    mapping: Dict[int, str] = {}
    df = _safe_read_csv(MEDICINE_METADATA_CSV)
    if df is None:
        _MEDICINE_NAME_MAP_CACHE = mapping
        return mapping

    id_col = None
    name_col = None
    for candidate in ["VAIPE2022_Class_ID", "class_id", "Class_ID"]:
        if candidate in df.columns:
            id_col = candidate
            break
    for candidate in ["Medicine Name", "medicine_name", "Medicine_Name"]:
        if candidate in df.columns:
            name_col = candidate
            break

    if id_col is None or name_col is None:
        _MEDICINE_NAME_MAP_CACHE = mapping
        return mapping

    for _, row in df.iterrows():
        try:
            cid = int(row[id_col])
        except (TypeError, ValueError):
            continue
        name = str(row.get(name_col, "")).strip()
        if name:
            mapping[cid] = name

    if 107 not in mapping:
        mapping[107] = "OUT_OF_PRESCRIPTION"

    _MEDICINE_NAME_MAP_CACHE = mapping
    return mapping


def _load_prescription_index() -> Optional[pd.DataFrame]:
    global _PRESCRIPTION_INDEX_CACHE
    if _PRESCRIPTION_INDEX_CACHE is not None:
        return _PRESCRIPTION_INDEX_CACHE

    df = _safe_read_csv(PRESCRIPTION_INDEX_CSV)
    if df is None or "prescription_image" not in df.columns:
        _PRESCRIPTION_INDEX_CACHE = None
        return None

    cached = df.copy()
    cached["__norm"] = cached["prescription_image"].astype(str).map(_normalize_path)
    cached["__base"] = cached["__norm"].map(lambda x: Path(x).name)
    _PRESCRIPTION_INDEX_CACHE = cached
    return cached


def _find_prescription_context(image_ref: str) -> Dict[str, Any]:
    df = _load_prescription_index()
    if df is None:
        return {
            "found": False,
            "reason": "Prescription index CSV not found or invalid.",
            "classes_in_prescription": [],
            "prescription_json": "",
        }

    target_norm = _normalize_path(image_ref)
    target_base = Path(target_norm).name
    matches = df[(df["__norm"] == target_norm) | (df["__base"] == target_base)]

    if matches.empty:
        return {
            "found": False,
            "reason": "Prescription image was not found in Prescription_Image_Index.csv.",
            "classes_in_prescription": [],
            "prescription_json": "",
        }

    row = matches.iloc[0]
    classes = _parse_prescription_class_ids(row.get("class_ids_in_prescription", ""))
    return {
        "found": True,
        "reason": "Matched by image filename/path in Prescription_Image_Index.csv.",
        "classes_in_prescription": classes,
        "prescription_json": str(row.get("prescription_json", "")).strip(),
        "matched_prescription_image": str(row.get("prescription_image", "")).strip(),
    }


def _get_model(model_name: str) -> Tuple[torch.nn.Module, Dict[int, str], Path]:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unsupported model name: {model_name}")

    ckpt_path = resolve_model_checkpoint_path(MODELS_DIR, model_name)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for {model_name}")

    key = (model_name, str(ckpt_path.resolve()), str(DEVICE))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    ckpt_class_to_idx = load_checkpoint_class_to_idx(str(ckpt_path), map_location=DEVICE)
    if not ckpt_class_to_idx:
        raise RuntimeError(
            f"Checkpoint {ckpt_path} does not contain class_to_idx. "
            "Please retrain or use checkpoint with mapping."
        )

    num_classes = len(ckpt_class_to_idx)
    model = load_checkpoint(
        model_name=model_name,
        num_classes=num_classes,
        checkpoint_path=str(ckpt_path),
        map_location=DEVICE,
    ).to(DEVICE)
    model.eval()

    inv_map = {int(v): str(k) for k, v in ckpt_class_to_idx.items()}
    payload = (model, inv_map, ckpt_path)
    _MODEL_CACHE[key] = payload
    return payload


def _classify_image(image_path: Path, model_name: str, top_k: int = 3) -> Dict[str, Any]:
    return _classify_images([image_path], model_name=model_name, top_k=top_k)[0]


def _classify_images(image_paths: Sequence[Path], model_name: str, top_k: int = 3) -> List[Dict[str, Any]]:
    if not image_paths:
        return []

    model, inv_map, ckpt_path = _get_model(model_name)
    med_map = _load_medicine_name_map()

    tensors: List[torch.Tensor] = []
    for image_path in image_paths:
        img = focus_on_object(pil_loader(str(image_path)), scale=0.85)
        tensors.append(_TRANSFORM(img))

    batch = torch.stack(tensors, dim=0).to(DEVICE)

    with torch.no_grad():
        use_amp = DEVICE.type == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(batch)
            probs = F.softmax(logits, dim=1).detach().cpu()

    k = max(1, min(int(top_k), int(probs.shape[1])))

    outputs: List[Dict[str, Any]] = []
    for prob_row in probs:
        top_probs, top_idxs = torch.topk(prob_row, k=k)

        top_predictions: List[Dict[str, Any]] = []
        for idx, prob in zip(top_idxs.tolist(), top_probs.tolist()):
            class_name = inv_map.get(int(idx), f"class_{int(idx):03d}")
            class_id = _parse_class_id(class_name)
            medicine_name = med_map.get(class_id, class_name) if class_id is not None else class_name
            top_predictions.append(
                {
                    "class_index": int(idx),
                    "class_name": class_name,
                    "class_id": class_id,
                    "medicine_name": medicine_name,
                    "confidence": float(prob),
                }
            )

        predicted = top_predictions[0]
        outputs.append(
            {
                "model_name": model_name,
                "device": str(DEVICE),
                "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "predicted": predicted,
                "top_k": top_predictions,
            }
        )

    return outputs


def _split_overview(root: Path, split: str) -> Dict[str, Any]:
    split_path = root / split
    if not split_path.exists():
        return {"exists": False, "class_count": 0, "total_images": 0, "empty_classes": []}

    class_dirs = sorted([d for d in split_path.iterdir() if d.is_dir()])
    empty_classes: List[str] = []
    total_images = 0

    for cls in class_dirs:
        n = sum(
            1
            for f in cls.rglob("*")
            if f.is_file() and f.suffix.lower() in ALLOWED_EXTS
        )
        total_images += n
        if n == 0:
            empty_classes.append(cls.name)

    return {
        "exists": True,
        "class_count": len(class_dirs),
        "total_images": total_images,
        "empty_classes": empty_classes,
        "class_names": [d.name for d in class_dirs],
    }


def _dataset_overview() -> Dict[str, Any]:
    root = DATA_ALIGNED_DIR
    train_info = _split_overview(root, "train")
    val_info = _split_overview(root, "val")
    test_info = _split_overview(root, "test")

    train_set = set(train_info.get("class_names", []))
    val_set = set(val_info.get("class_names", []))
    test_set = set(test_info.get("class_names", []))

    return {
        "data_aligned_exists": root.exists(),
        "class_sets_equal": train_set == val_set == test_set,
        "train": train_info,
        "val": val_info,
        "test": test_info,
    }


def _model_overview() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name in MODEL_NAMES:
        ckpt = resolve_model_checkpoint_path(MODELS_DIR, name)
        exists = ckpt is not None and ckpt.exists()
        class_count = None
        if exists and ckpt is not None:
            c2i = load_checkpoint_class_to_idx(str(ckpt), map_location="cpu")
            if isinstance(c2i, dict):
                class_count = len(c2i)

        rows.append(
            {
                "model_name": name,
                "checkpoint_exists": bool(exists),
                "checkpoint": str(ckpt.relative_to(PROJECT_ROOT)).replace("\\", "/") if exists and ckpt else "",
                "class_count": class_count,
            }
        )
    return rows


def _safe_df_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def _overview_payload(force_refresh: bool = False) -> Dict[str, Any]:
    global _OVERVIEW_CACHE_DATA, _OVERVIEW_CACHE_EXPIRES_AT

    now = time.monotonic()
    if (
        not force_refresh
        and _OVERVIEW_CACHE_DATA is not None
        and now < _OVERVIEW_CACHE_EXPIRES_AT
    ):
        return _OVERVIEW_CACHE_DATA

    ann_df = _safe_read_csv(PRESCRIPTION_ANN_CSV)
    ann_stats: Dict[str, Any] = {"rows": 0, "target_class_min": None, "target_class_max": None}
    if ann_df is not None and "target_class_id" in ann_df.columns:
        series = pd.to_numeric(ann_df["target_class_id"], errors="coerce").dropna().astype(int)
        if not series.empty:
            ann_stats = {
                "rows": int(len(ann_df)),
                "target_class_min": int(series.min()),
                "target_class_max": int(series.max()),
                "target_class_unique": int(series.nunique()),
                "contains_107": bool((series == 107).any()),
            }

    payload = {
        "dataset": _dataset_overview(),
        "models": _model_overview(),
        "evaluation_summary": _safe_df_records(EVAL_SUMMARY_CSV),
        "training_table": _safe_df_records(TRAIN_TABLE_CSV),
        "prescription_annotation_stats": ann_stats,
        "cache": {
            "overview_cache_ttl_sec": OVERVIEW_CACHE_TTL_SEC,
            "generated_at_unix": int(time.time()),
        },
    }

    _OVERVIEW_CACHE_DATA = payload
    _OVERVIEW_CACHE_EXPIRES_AT = now + OVERVIEW_CACHE_TTL_SEC
    return payload


def _api_error(message: str, status: int = 400):
    return jsonify({"ok": False, "error": message}), status


app = Flask(
    __name__,
    template_folder=str(FRONTEND_ROOT / "templates"),
    static_folder=str(FRONTEND_ROOT / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("THUOC_WEB_MAX_UPLOAD_MB", "16")) * 1024 * 1024


@app.errorhandler(RequestEntityTooLarge)
def handle_request_entity_too_large(_exc):
    max_mb = int(os.getenv("THUOC_WEB_MAX_UPLOAD_MB", "16"))
    return _api_error(f"Uploaded file is too large. Max size is {max_mb} MB.", status=413)


@app.get("/")
def home():
    return render_template(
        "index.html",
        model_names=MODEL_NAMES,
        default_model="efficientnet_b0",
        device=str(DEVICE),
        max_pill_images=MAX_PILL_IMAGES,
    )


@app.get("/api/health")
def health():
    return jsonify(
        {
            "ok": True,
            "service": "thuoc-web",
            "device": str(DEVICE),
            "project_root": str(PROJECT_ROOT).replace("\\", "/"),
        }
    )


@app.get("/api/overview")
def api_overview():
    force_refresh = _is_truthy(request.args.get("force", "0"))
    try:
        return jsonify({"ok": True, "data": _overview_payload(force_refresh=force_refresh)})
    except Exception as exc:  # pragma: no cover - defensive
        return _api_error(f"Failed to build overview: {exc}", status=500)


@app.post("/api/classify")
def api_classify():
    image = request.files.get("image")
    if image is None:
        return _api_error("Missing file field 'image'.")

    model_name = str(request.form.get("model_name", "efficientnet_b0")).strip() or "efficientnet_b0"
    if model_name not in MODEL_NAMES:
        return _api_error(f"Unsupported model_name: {model_name}")

    top_k = _parse_top_k(request.form.get("top_k", "3"))

    try:
        image_path, original_name, stored_name = _save_uploaded_file(image, UPLOAD_PILLS)
        result = _classify_image(image_path=image_path, model_name=model_name, top_k=top_k)
        result["uploaded_file"] = {
            "original_name": original_name,
            "stored_name": stored_name,
            "saved_path": str(image_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        }
        return jsonify({"ok": True, "data": result})
    except ValueError as exc:
        return _api_error(str(exc), status=400)
    except FileNotFoundError as exc:
        return _api_error(str(exc), status=404)
    except Exception as exc:
        return _api_error(str(exc), status=500)


@app.post("/api/check-prescription")
def api_check_prescription():
    prescription_image = request.files.get("prescription_image")
    if prescription_image is None:
        return _api_error("Missing file field 'prescription_image'.")

    pill_files = request.files.getlist("pill_images")
    if not pill_files:
        maybe_single = request.files.get("pill_image")
        if maybe_single is not None:
            pill_files = [maybe_single]

    if not pill_files:
        return _api_error("Missing file field 'pill_images' (at least one image is required).")

    model_name = str(request.form.get("model_name", "efficientnet_b0")).strip() or "efficientnet_b0"
    if model_name not in MODEL_NAMES:
        return _api_error(f"Unsupported model_name: {model_name}")

    if len(pill_files) > MAX_PILL_IMAGES:
        return _api_error(
            f"Too many pill images. Maximum allowed is {MAX_PILL_IMAGES} images per request."
        )

    top_k = _parse_top_k(request.form.get("top_k", "3"))
    use_annotation_lookup = str(request.form.get("use_annotation_lookup", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
    }

    try:
        presc_path, presc_original_name, presc_stored_name = _save_uploaded_file(
            prescription_image,
            UPLOAD_PRESCRIPTIONS,
        )

        # Try matching against both uploaded display name and stored name.
        context = _find_prescription_context(presc_original_name)
        if not context.get("found", False):
            context = _find_prescription_context(presc_stored_name)
        if not context.get("found", False):
            # Fallback: try the saved filename/path as reference.
            context = _find_prescription_context(str(presc_path))

        classes_in_prescription = set(context.get("classes_in_prescription", []))

        uploaded_pills: List[Dict[str, Any]] = []
        pill_paths: List[Path] = []
        pill_original_names: List[str] = []
        pill_stored_names: List[str] = []
        for pill_file in pill_files:
            pill_path, pill_original_name, pill_stored_name = _save_uploaded_file(pill_file, UPLOAD_PILLS)
            pill_paths.append(pill_path)
            pill_original_names.append(pill_original_name)
            pill_stored_names.append(pill_stored_name)
            uploaded_pills.append(
                {
                    "path": pill_path,
                    "original_name": pill_original_name,
                    "stored_name": pill_stored_name,
                }
            )

        classify_results = _classify_images(pill_paths, model_name=model_name, top_k=top_k)

        items: List[Dict[str, Any]] = []
        for upload_info, classify_result in zip(uploaded_pills, classify_results):
            pred = classify_result["predicted"]
            pred_class_id = pred.get("class_id")

            is_in: Optional[bool] = None
            is_out: Optional[bool] = None
            if context.get("found", False) and pred_class_id is not None:
                is_in = int(pred_class_id) in classes_in_prescription
                is_out = not is_in

            items.append(
                {
                    "pill_file": {
                        "original_name": upload_info["original_name"],
                        "stored_name": upload_info["stored_name"],
                        "saved_path": str(upload_info["path"].relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    },
                    "classification": classify_result,
                    "is_in_prescription": is_in,
                    "is_out_of_prescription": is_out,
                }
            )

        has_out: Optional[bool] = None
        all_in: Optional[bool] = None
        if context.get("found", False):
            flags = [x["is_out_of_prescription"] for x in items if x["is_out_of_prescription"] is not None]
            has_out = any(flags) if flags else False
            all_in = not has_out

        annotation_lookup_payload: Optional[Dict[str, Any]] = None
        if use_annotation_lookup:
            try:
                lookup_result = match_pills_to_prescription(
                    prescription_image=presc_original_name,
                    pill_images=pill_original_names,
                    annotations_csv=PRESCRIPTION_ANN_CSV,
                    prescription_index_csv=PRESCRIPTION_INDEX_CSV,
                    metadata_csv=MEDICINE_METADATA_CSV,
                )
                annotation_lookup_payload = result_to_dict(lookup_result)
            except Exception as lookup_exc:
                # Fallback when uploaded filenames are sanitized/renamed by client.
                try:
                    lookup_result = match_pills_to_prescription(
                        prescription_image=presc_stored_name,
                        pill_images=pill_stored_names,
                        annotations_csv=PRESCRIPTION_ANN_CSV,
                        prescription_index_csv=PRESCRIPTION_INDEX_CSV,
                        metadata_csv=MEDICINE_METADATA_CSV,
                    )
                    annotation_lookup_payload = result_to_dict(lookup_result)
                    annotation_lookup_payload["fallback_used"] = "stored_name"
                    annotation_lookup_payload["initial_error"] = str(lookup_exc)
                except Exception as fallback_exc:
                    annotation_lookup_payload = {
                        "error": str(lookup_exc),
                        "fallback_error": str(fallback_exc),
                    }

        payload = {
            "model_name": model_name,
            "device": str(DEVICE),
            "prescription_file": {
                "original_name": presc_original_name,
                "stored_name": presc_stored_name,
                "saved_path": str(presc_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            },
            "prescription_context": context,
            "items": items,
            # TRUE means all uploaded pill images belong to prescription list.
            # FALSE means at least one pill is outside prescription.
            "analysis_true_false": all_in,
            "has_out_of_prescription": has_out,
            "annotation_lookup": annotation_lookup_payload,
        }
        return jsonify({"ok": True, "data": payload})
    except ValueError as exc:
        return _api_error(str(exc), status=400)
    except FileNotFoundError as exc:
        return _api_error(str(exc), status=404)
    except Exception as exc:
        return _api_error(str(exc), status=500)


def main() -> None:
    host = os.getenv("THUOC_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("THUOC_WEB_PORT", "5000"))
    debug = os.getenv("THUOC_WEB_DEBUG", "0") == "1"

    UPLOAD_PRESCRIPTIONS.mkdir(parents=True, exist_ok=True)
    UPLOAD_PILLS.mkdir(parents=True, exist_ok=True)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
