import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import tournament  # noqa: E402,F401  (adds backend/ to sys.path)

from app.chess_net.config import resolve_config  # noqa: E402
from app.chess_net.model import build_model  # noqa: E402
from app.chess_net.model_io import save_model  # noqa: E402


def make_fake_model(out_dir: Path, seed: int = 0, nickname: str = "tester", model_name: str | None = None, nan: bool = False,
                    raw: dict | None = None) -> Path:
    """A tiny untrained network saved in the real model format (SPEC 8.4)."""
    cfg, err = resolve_config({"layers": 2, "width": 16, "seed": seed, **(raw or {})})
    assert not err, err
    torch.manual_seed(seed)
    model = build_model(cfg)
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    if nan:
        sd["out.bias"] = torch.full_like(sd["out.bias"], float("nan"))
    save_model(out_dir, sd, cfg, {"stop_reason": "budget", "val_mse": 0.1}, [],
               {"nickname": nickname, "model_name": model_name or f"net{seed}"})
    return out_dir


@pytest.fixture
def models_dir(tmp_path):
    d = tmp_path / "models"
    for i in range(4):
        make_fake_model(d / f"sub{i}", seed=i, nickname=f"p{i}", model_name=f"Net{i}")
    return d
