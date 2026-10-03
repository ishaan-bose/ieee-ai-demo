#!/usr/bin/env python3
"""Export the trained showcase models to browser-loadable files (SPEC 9): frontend/public/models/<name>.bin + manifest.json.

Reads STATE_DIR/showcase/{doodle,raw_audio,logmel_audio}.pt (written by train_showcase.py). Idempotent: rewrites files only when
the .pt is newer. The TypeScript loader is frontend/src/lib/inference.ts; the format is documented in app/export_format.py.

    cd ~/ieee-ai-demo/backend && python3 scripts/export_weights.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO = Path(__file__).resolve().parents[2]

from app.config import get_settings  # noqa: E402
from app.data import audio_features as af  # noqa: E402
from app.export_format import export_sequential  # noqa: E402
from app.training import audio_models as am  # noqa: E402
from app.training.classifier import build_mlp_classifier  # noqa: E402
from scripts._common import setup_logging, write_json  # noqa: E402
from scripts.train_showcase import DOODLE_HIDDEN  # noqa: E402

import torch  # noqa: E402

MODELS = {
    "doodle": {"build": lambda: build_mlp_classifier(784, DOODLE_HIDDEN, 10, "relu"),
               "input": {"kind": "doodle", "shape": [784], "note": "28x28 uint8 raster / 255, flattened row-major"}},
    "raw_audio": {"build": lambda: am.build_raw_model(),
                  "input": {"kind": "raw_audio", "shape": [af.RAW_LEN], "note": "int16/32768 average-pooled by 4 (AUDIO_FEATURES.md)"}},
    "logmel_audio": {"build": lambda: am.build_logmel_model(),
                     "input": {"kind": "logmel", "shape": [1, af.N_MELS, af.N_FRAMES], "note": "log-mel (AUDIO_FEATURES.md), mel x frame"}},
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frontend-dir", type=Path, default=REPO / "frontend")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    s = get_settings()
    log = setup_logging("export_weights", s)
    src = s.state_dir / "showcase"
    out = a.frontend_dir / "public" / "models"
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {"version": 1, "models": {}}
    summary = {}
    for name, spec in MODELS.items():
        pt = src / f"{name}.pt"
        if not pt.is_file():
            log.warning("%s: %s not found (run train_showcase.py first); skipping", name, pt)
            continue
        binf = out / f"{name}.bin"
        if binf.is_file() and not a.force and binf.stat().st_mtime >= pt.stat().st_mtime and name in manifest["models"]:
            log.info("%s: up to date", name)
            summary[name] = "up to date"
            continue
        ck = torch.load(pt, map_location="cpu", weights_only=False)
        model: nn.Sequential = spec["build"]()
        model.load_state_dict(ck["state_dict"])
        layers, weights = export_sequential(model)
        weights.astype("<f4").tofile(binf)
        m = ck.get("metrics", {})
        manifest["models"][name] = {
            "file": f"models/{name}.bin", "dtype": "float32", "floats": int(weights.size), "layers": layers, "input": spec["input"],
            "classes": ck.get("classes"), "params": int(weights.size),
            "val_acc": m.get("val_acc"), "val_acc_partial": m.get("val_acc_partial"), "test_acc": m.get("test_acc")}
        log.info("%s: %d floats (%.2f MB), val acc %s", name, weights.size, weights.size * 4 / 1e6, m.get("val_acc"))
        summary[name] = {"floats": int(weights.size), "val_acc": m.get("val_acc")}
    write_json(manifest_path, manifest)
    print("SUMMARY export_weights:", json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
