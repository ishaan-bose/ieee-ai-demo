import os
from pathlib import Path

import pytest

from app.config import HOST, get_settings


def test_defaults(monkeypatch):
    for var in ("DATA_DIR", "STATE_DIR", "NUM_WORKERS", "BACKEND_PORT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("app.config.load_env_file", lambda path: None)  # ignore a local backend/.env
    s = get_settings()
    assert s.data_dir == Path("~/demo/data").expanduser().resolve()
    assert s.num_workers == 4
    assert s.host == HOST == "127.0.0.1"
    assert s.port == 8000


def test_num_workers_never_from_cpu_count(monkeypatch):
    monkeypatch.delenv("NUM_WORKERS", raising=False)
    monkeypatch.setattr("app.config.load_env_file", lambda path: None)
    monkeypatch.setattr(os, "cpu_count", lambda: 256)  # what the server reports
    assert get_settings().num_workers == 4


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("NUM_WORKERS", "7")
    s = get_settings()
    assert s.data_dir == (tmp_path / "d").resolve()
    assert s.num_workers == 7


@pytest.mark.parametrize("bad", ["0", "-1", "four"])
def test_bad_num_workers(monkeypatch, bad):
    monkeypatch.setenv("NUM_WORKERS", bad)
    with pytest.raises(ValueError):
        get_settings()


def test_env_file_does_not_override_real_env(monkeypatch, tmp_path):
    from app.config import load_env_file

    f = tmp_path / ".env"
    f.write_text("# comment\nFOO_TEST_A=from_file\nexport FOO_TEST_B='quoted'\n")
    monkeypatch.setenv("FOO_TEST_A", "from_env")
    monkeypatch.delenv("FOO_TEST_B", raising=False)
    load_env_file(f)
    assert os.environ["FOO_TEST_A"] == "from_env"
    assert os.environ["FOO_TEST_B"] == "quoted"
    monkeypatch.delenv("FOO_TEST_B")
