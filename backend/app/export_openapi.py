"""Write shared/openapi.json from the FastAPI app (SPEC 7: one source for frontend types).

Run from backend/:  python -m app.export_openapi
"""

from __future__ import annotations

import json
from pathlib import Path

from app.main import app

OUT = Path(__file__).resolve().parents[2] / "shared" / "openapi.json"


def main() -> None:
    OUT.write_text(json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
