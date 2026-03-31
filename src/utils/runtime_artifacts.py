from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, Tuple


def ensure_runtime_dirs(log_dir: str | Path = "log", json_dir: str | Path = "json") -> Tuple[Path, Path]:
    log_path = Path(log_dir)
    json_path = Path(json_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    json_path.mkdir(parents=True, exist_ok=True)
    return log_path, json_path


def mirror_artifacts_to_runtime_dirs(
    source_roots: Iterable[str | Path],
    *,
    log_dir: str | Path = "log",
    json_dir: str | Path = "json",
) -> dict[str, int]:
    """Copy .log and .json artifacts into dedicated runtime folders, preserving relative tree."""
    log_root, json_root = ensure_runtime_dirs(log_dir=log_dir, json_dir=json_dir)

    copied_log = 0
    copied_json = 0
    log_root_resolved = log_root.resolve()
    json_root_resolved = json_root.resolve()

    for root_like in source_roots:
        root = Path(root_like)
        if not root.exists():
            continue

        if root.is_file():
            files = [root]
            root_base = root.parent
        else:
            files = [p for p in root.rglob("*") if p.is_file()]
            root_base = root

        for src in files:
            src_resolved = src.resolve()
            if src_resolved.is_relative_to(log_root_resolved) or src_resolved.is_relative_to(json_root_resolved):
                continue

            suffix = src.suffix.lower()
            if suffix not in {".log", ".json"}:
                continue

            try:
                rel = src.relative_to(root_base)
            except ValueError:
                rel = Path(src.name)

            if suffix == ".log":
                dst = log_root / root.name / rel
                copied_log += 1
            else:
                dst = json_root / root.name / rel
                copied_json += 1

            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    return {"copied_log": copied_log, "copied_json": copied_json}

