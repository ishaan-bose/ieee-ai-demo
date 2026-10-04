"""In-memory ring buffer of recent log lines for GET /admin/logs and the admin live monitor, plus a rotating log file."""

from __future__ import annotations

import collections
import logging
import logging.handlers
import threading
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class RingHandler(logging.Handler):
    def __init__(self, capacity: int = 2000):
        super().__init__()
        self.lines: collections.deque[str] = collections.deque(maxlen=capacity)
        self._lock2 = threading.Lock()
        self.setFormatter(logging.Formatter(_FORMAT, "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            with self._lock2:
                self.lines.append(self.format(record))
        except Exception:  # noqa: BLE001
            pass

    def tail(self, n: int) -> list[str]:
        with self._lock2:
            return list(self.lines)[-n:]


_ring: RingHandler | None = None


def install(log_dir: Path | None = None) -> RingHandler:
    """Attach the ring (and a rotating file in log_dir) to the 'app' and 'uvicorn' loggers. Idempotent."""
    global _ring
    if _ring is None:
        _ring = RingHandler()
        for name in ("app", "scheduler", "uvicorn.error"):
            lg = logging.getLogger(name)
            lg.setLevel(logging.INFO)
            lg.addHandler(_ring)
            lg.propagate = name != "app" and name != "scheduler"
        if log_dir is not None:
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                fh = logging.handlers.RotatingFileHandler(log_dir / "backend.log", maxBytes=5_000_000, backupCount=3)
                fh.setFormatter(logging.Formatter(_FORMAT))
                for name in ("app", "scheduler"):
                    logging.getLogger(name).addHandler(fh)
            except OSError:
                pass
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(_FORMAT, "%H:%M:%S"))
        for name in ("app", "scheduler"):
            logging.getLogger(name).addHandler(sh)
    return _ring


def tail(n: int = 200) -> list[str]:
    return _ring.tail(n) if _ring else []
