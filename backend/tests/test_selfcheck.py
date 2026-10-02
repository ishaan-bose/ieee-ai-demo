from app.config import get_settings
from app.device import DeviceInfo, detect_device
from app.selfcheck import run_self_check, smoke_test
from tests.conftest import requires_torch


def test_detect_device():
    d = detect_device()
    assert d.device in ("cpu", "cuda")
    if d.device == "cuda":
        assert d.gpu_name and d.vram_gb and d.vram_gb > 0


@requires_torch
def test_smoke_test_cpu():
    r = smoke_test("cpu")
    assert r["ok"], r


@requires_torch
def test_self_check_on_fake_data(app_env):
    lines = []
    r = run_self_check(get_settings(), detect_device(), print_fn=lines.append)
    assert r.ok and not r.warnings
    assert r.datasets["lichess"]["boards.npy"]["dtype"] == "uint8"
    assert r.datasets["speech"]["X_train.npy"]["dtype"] == "int16"
    assert "not built yet" in r.datasets["quickdraw"]["tensors"]
    text = "\n".join(lines)
    assert "SELF-CHECK" in text and "smoke test  : PASS" in text


@requires_torch
def test_self_check_missing_data_does_not_crash(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "nothing_here"))
    cpu = DeviceInfo("cpu", None, None, None, None, None)
    r = run_self_check(get_settings(), cpu, print_fn=lambda s: None)
    assert r.smoke_test["ok"]  # smoke test is independent of data
    assert any("missing" in w for w in r.warnings)
