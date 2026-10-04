"""Tournament CLI (SPEC 10): separate from the API server, which never imports or starts it.

Run from the repo root:  python -m tournament.run --rounds 6 --out results/
The model loader and the position encoder are imported from backend/app (ONE implementation, shared with training).
"""

import sys
from pathlib import Path

_BACKEND = str(Path(__file__).resolve().parents[1] / "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
