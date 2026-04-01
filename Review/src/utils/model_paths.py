from __future__ import annotations

from pathlib import Path
from typing import Optional

# Keep folder naming aligned with current repository layout under models/AI/.
_MODEL_DIR_ALIASES: dict[str, str] = {
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet",
    "vit_b_16": "vit_b_16",
}


def _alias_of(model_name: str) -> str:
    return _MODEL_DIR_ALIASES.get(str(model_name), str(model_name))


def _is_model_artifact_dir(path: Path, model_name: str) -> bool:
    alias = _alias_of(model_name)
    return path.name == alias and path.parent.name == "AI"


def model_artifact_dir(base_output_dir: str | Path, model_name: str) -> Path:
    root = Path(base_output_dir)
    alias = _alias_of(model_name)

    if _is_model_artifact_dir(root, model_name):
        return root
    if root.name == "AI":
        return root / alias
    return root / "AI" / alias


def single_mode_output_dir(base_output_dir: str | Path, model_name: str) -> Path:
    """Return per-model artifact dir, but keep caller-provided model dir unchanged."""
    return model_artifact_dir(base_output_dir, model_name)


def resolve_model_checkpoint_path(base_output_dir: str | Path, model_name: str) -> Optional[Path]:
    root = Path(base_output_dir)
    ckpt_name = f"{model_name}_epillid_best.pt"
    alias = _alias_of(model_name)
    preferred = model_artifact_dir(root, model_name) / ckpt_name

    direct = root / ckpt_name
    alias_direct = root / alias / ckpt_name
    ai_direct = (root / "AI" / alias / ckpt_name) if root.name != "AI" else (root / alias / ckpt_name)

    candidates: list[Path] = []
    for candidate in [preferred, direct, alias_direct, ai_direct]:
        if candidate.exists():
            candidates.append(candidate)

    search_roots = {root}
    if root.name == "AI":
        search_roots.add(root.parent)
    else:
        search_roots.add(root / "AI")
    if _is_model_artifact_dir(root, model_name):
        search_roots.add(root.parent)
        search_roots.add(root.parent.parent)

    for search_root in search_roots:
        if not search_root.exists():
            continue
        for candidate in sorted(search_root.glob(f"**/{ckpt_name}")):
            if candidate.exists():
                candidates.append(candidate)

    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(candidate.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(candidate)

    if not unique_candidates:
        return None

    preferred_resolved = str(preferred.resolve())

    def _rank(path: Path) -> tuple[int, float]:
        is_preferred = int(str(path.resolve()) == preferred_resolved)
        return (is_preferred, path.stat().st_mtime)

    return max(unique_candidates, key=_rank)


def resolve_model_artifact_path(
    base_output_dir: str | Path,
    model_name: str,
    suffix: str,
) -> Optional[Path]:
    ckpt = resolve_model_checkpoint_path(base_output_dir=base_output_dir, model_name=model_name)
    if ckpt is None:
        return None
    return ckpt.with_name(f"{model_name}{suffix}")

