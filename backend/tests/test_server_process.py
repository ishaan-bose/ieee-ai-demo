"""Runs the real server process (python -m app.main) like the owner does."""

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from tests.conftest import HAS_TORCH

BACKEND_DIR = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server(tmp_path, fake_data_dir):
    port = _free_port()
    env = {**os.environ, "DATA_DIR": str(fake_data_dir), "STATE_DIR": str(tmp_path / "state"),
           "BACKEND_PORT": str(port), "SELF_CHECK": "1"}
    proc = subprocess.Popen([sys.executable, "-m", "app.main"], cwd=BACKEND_DIR, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            httpx.get(f"{base}/api/health", timeout=0.5)
            break
        except httpx.TransportError:
            time.sleep(0.2)
    else:
        proc.kill()
        pytest.fail("server did not start:\n" + proc.communicate()[0])
    yield proc, base, port
    if proc.poll() is None:
        proc.kill()
        proc.wait()


@pytest.mark.skipif(not Path("/proc/net/tcp").exists(), reason="Linux only")
def test_binds_localhost_only(server):
    proc, base, port = server
    assert httpx.get(f"{base}/api/health").json()["ok"] is True
    # The listening socket must be 127.0.0.1, not 0.0.0.0 (SPEC 13).
    listening = [l for l in Path("/proc/net/tcp").read_text().splitlines()[1:] if l.split()[3] == "0A"]
    ours = [l.split()[1] for l in listening if l.split()[1].endswith(f":{port:04X}")]
    assert ours == [f"0100007F:{port:04X}"], ours


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM], ids=["ctrl-c", "kill"])
def test_shutdown_does_not_hang_on_open_stream(server, sig):
    """Regression: uvicorn waited forever for open SSE streams on Ctrl+C / restart."""
    proc, base, _ = server
    with httpx.stream("GET", f"{base}/api/dev/hello-stream", timeout=10) as r:
        lines = r.iter_lines()
        for _ in range(4):  # two ticks: the stream is open and the generator is mid-sleep
            next(lines)
        proc.send_signal(sig)
        t0 = time.time()
        proc.wait(timeout=15)
    assert time.time() - t0 < 10
    out = proc.stdout.read()
    assert "SELF-CHECK" in out
    if HAS_TORCH:
        assert "smoke test  : PASS" in out
