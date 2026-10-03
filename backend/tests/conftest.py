"""Shared fixtures. Tests never touch the real ~/demo paths: DATA_DIR and STATE_DIR
point at temporary directories filled with fake data."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fake_data import make_fake_data

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = REPO_ROOT / "data" / "samples"

# torch is preinstalled on the server; a laptop may not have it. Tests that need the
# training smoke test skip instead of failing there.
try:
    import torch  # noqa: F401

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


@pytest.fixture(scope="session")
def fake_data_dir(tmp_path_factory) -> Path:
    return make_fake_data(tmp_path_factory.mktemp("fake_data"), n=60, seed=1)


@pytest.fixture
def app_env(monkeypatch, tmp_path, fake_data_dir):
    """Environment for starting the app on fake data with a fresh database."""
    monkeypatch.setenv("DATA_DIR", str(fake_data_dir))
    monkeypatch.setenv("STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("NUM_WORKERS", "2")
    monkeypatch.setenv("SELF_CHECK", "1")
    return tmp_path / "state"


@pytest.fixture
def client(app_env):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:  # runs the lifespan (db init, device detection, self-check)
        yield c
