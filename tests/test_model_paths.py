from pathlib import Path

from src.utils.model_paths import (
    model_artifact_dir,
    resolve_model_checkpoint_path,
    single_mode_output_dir,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"ok")


def test_model_artifact_dir_supports_models_and_models_ai_roots() -> None:
    assert model_artifact_dir("models", "resnet50") == Path("models") / "AI" / "resnet50"
    assert model_artifact_dir("models/AI", "resnet50") == Path("models") / "AI" / "resnet50"
    assert model_artifact_dir("models/AI/efficientnet", "efficientnet_b0") == Path("models") / "AI" / "efficientnet"


def test_single_mode_output_dir_keeps_canonical_alias() -> None:
    assert single_mode_output_dir("models", "efficientnet_b0") == Path("models") / "AI" / "efficientnet"
    assert single_mode_output_dir("models/AI", "efficientnet_b0") == Path("models") / "AI" / "efficientnet"
    assert single_mode_output_dir("models/AI/efficientnet", "efficientnet_b0") == Path("models") / "AI" / "efficientnet"


def test_resolve_model_checkpoint_path_finds_checkpoint_from_models_root(tmp_path: Path) -> None:
    ckpt = tmp_path / "models" / "AI" / "vit_b_16" / "vit_b_16_epillid_best.pt"
    _touch(ckpt)

    resolved = resolve_model_checkpoint_path(tmp_path / "models", "vit_b_16")
    assert resolved is not None
    assert resolved.resolve() == ckpt.resolve()


def test_resolve_model_checkpoint_path_finds_checkpoint_from_ai_root(tmp_path: Path) -> None:
    ckpt = tmp_path / "models" / "AI" / "resnet50" / "resnet50_epillid_best.pt"
    _touch(ckpt)

    resolved = resolve_model_checkpoint_path(tmp_path / "models" / "AI", "resnet50")
    assert resolved is not None
    assert resolved.resolve() == ckpt.resolve()


def test_resolve_model_checkpoint_path_finds_checkpoint_from_model_dir(tmp_path: Path) -> None:
    ckpt = tmp_path / "models" / "AI" / "efficientnet" / "efficientnet_b0_epillid_best.pt"
    _touch(ckpt)

    resolved = resolve_model_checkpoint_path(tmp_path / "models" / "AI" / "efficientnet", "efficientnet_b0")
    assert resolved is not None
    assert resolved.resolve() == ckpt.resolve()

