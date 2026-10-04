"""Runtime configuration, read from environment variables (SPEC 0, 0.1).

Rules:
- DATA_DIR comes from the environment (default ~/demo/data). Never hardcode data paths.
- NUM_WORKERS comes from the environment (default 4). Never size anything from
  os.cpu_count() or nproc: on the server they report the physical host (256 CPUs).
- The backend only ever binds 127.0.0.1 (SPEC 13).

An optional `backend/.env` file (KEY=VALUE lines, gitignored) is read at import time.
Values already set in the real environment win over the file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
HOST = "127.0.0.1"  # fixed on purpose: no public port, access is via the SSH tunnel


def load_env_file(path: Path) -> None:
    """Minimal .env reader: KEY=VALUE per line, '#' comments, optional quotes."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _env_path(name: str, default: str) -> Path:
    return Path(os.path.expanduser(os.environ.get(name) or default)).resolve()


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from e
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    state_dir: Path
    log_dir: Path
    num_workers: int
    port: int
    admin_token: str | None = field(repr=False)
    run_self_check: bool
    budget_flops: float = 1e16  # competition FLOPs budget (SPEC 6.2); calibrated by scripts/benchmark.py
    time_cap_seconds: float = 1620.0  # active-GPU-time safety net (~3x the nominal 9 minutes)
    checkpoint_seconds: float = 10.0  # checkpoint every ~this much active time
    metrics_seconds: float = 2.0
    val_rows: int = 50_000
    race_max_seconds: float = 120.0  # longest demo race a client may request
    host: str = HOST

    @property
    def calibration_path(self) -> Path:
        return self.state_dir / "calibration.json"

    @property
    def db_path(self) -> Path:
        return self.state_dir / "demo.db"

    @property
    def models_dir(self) -> Path:
        return self.state_dir / "models"

    @property
    def checkpoints_dir(self) -> Path:
        return self.state_dir / "checkpoints"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be a number, got {raw!r}") from e


def get_settings() -> Settings:
    """Read settings from the environment. Called at app startup (and by tests).

    BUDGET_FLOPS / TIME_CAP_SECONDS: env var wins, then STATE_DIR/calibration.json (written by benchmark.py), then defaults.
    """
    import json

    load_env_file(BACKEND_DIR / ".env")
    state_dir = _env_path("STATE_DIR", "~/demo/state")
    calib: dict = {}
    try:
        calib = json.loads((state_dir / "calibration.json").read_text())
    except (OSError, ValueError):
        pass
    return Settings(
        data_dir=_env_path("DATA_DIR", "~/demo/data"),
        state_dir=state_dir,
        log_dir=_env_path("LOG_DIR", "~/demo/logs"),
        num_workers=_env_int("NUM_WORKERS", 4),
        port=_env_int("BACKEND_PORT", 8000),
        admin_token=os.environ.get("ADMIN_TOKEN") or None,
        run_self_check=os.environ.get("SELF_CHECK", "1") not in ("0", "false", "no"),
        budget_flops=_env_float("BUDGET_FLOPS", float(calib.get("budget_flops", 1e16))),
        time_cap_seconds=_env_float("TIME_CAP_SECONDS", float(calib.get("time_cap_seconds", 1620.0))),
        checkpoint_seconds=_env_float("CHECKPOINT_SECONDS", 10.0),
        metrics_seconds=_env_float("METRICS_SECONDS", 2.0),
        val_rows=_env_int("VAL_ROWS", 50_000),
        race_max_seconds=_env_float("RACE_MAX_SECONDS", 120.0),
    )
