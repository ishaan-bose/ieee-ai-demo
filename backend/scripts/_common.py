"""Shared helpers for the server scripts: logging to ~/demo/logs (and the terminal), summaries."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def setup_logging(name: str, settings) -> logging.Logger:
    """Log to stdout and to LOG_DIR/<name>.log (appending)."""
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(settings.log_dir / f"{name}.log")):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.propagate = False
    return log


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, separators=(",", ":")))
    tmp.replace(path)


class Stopwatch:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self) -> float:
        return time.time() - self.t0
