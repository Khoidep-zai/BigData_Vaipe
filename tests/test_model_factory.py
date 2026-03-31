import torch

from src.models import model_factory


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loaded = None

    def load_state_dict(self, state_dict, strict=True):  # noqa: D401
        self.loaded = dict(state_dict)
        return self


def test_normalize_legacy_vit_head_keys() -> None:
    state_dict = {
        "heads.head.weight": torch.randn(3, 4),
        "heads.head.bias": torch.randn(3),
    }

    out = model_factory._normalize_legacy_state_dict("vit_b_16", dict(state_dict))

    assert "heads.head.weight" not in out
    assert "heads.head.bias" not in out
    assert "heads.head.1.weight" in out
    assert "heads.head.1.bias" in out


def test_load_checkpoint_vit_supports_legacy_head_layout(monkeypatch) -> None:
    fake_ckpt = {
        "model_state_dict": {
            "heads.head.weight": torch.randn(5, 6),
            "heads.head.bias": torch.randn(5),
        }
    }

    dummy_model = _DummyModel()

    monkeypatch.setattr(model_factory.torch, "load", lambda *args, **kwargs: fake_ckpt)
    monkeypatch.setattr(model_factory, "create_model", lambda *args, **kwargs: (dummy_model, 0))

    model = model_factory.load_checkpoint(
        model_name="vit_b_16",
        num_classes=None,
        checkpoint_path="dummy.pt",
        map_location="cpu",
    )

    assert model is dummy_model
    assert dummy_model.loaded is not None
    assert "heads.head.1.weight" in dummy_model.loaded
    assert "heads.head.1.bias" in dummy_model.loaded

