"""Regenerate shared/golden/*.json from data/samples/ (only needed if the specs change).

Run from backend/:  python3 scripts/make_golden.py
Tests check the committed fixtures against both Python and TypeScript.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data import audio_features as af  # noqa: E402
from app.data.rasterizer import rasterize  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "shared" / "golden"


def rasterizer_fixture() -> dict:
    qd = ROOT / "data" / "samples" / "quickdraw"
    samples = json.loads((qd / "samples.json").read_text())
    duel = json.loads((qd / "duel.json").read_text())
    cases = []
    for r in samples[::3] + duel[:6]:
        cases.append({"name": f"qd_{r['k']}", "strokes": r["d"], "pixels": rasterize(r["d"]).ravel().tolist()})
    # browser-style input: float canvas coordinates at a different scale/offset, a dot, a straight line
    rng = np.random.default_rng(0)
    pts = np.cumsum(rng.normal(0, 25, (40, 2)), axis=0) + 300
    browser = [[np.round(pts[:20, 0], 2).tolist(), np.round(pts[:20, 1], 2).tolist()],
               [np.round(pts[20:, 0], 2).tolist(), np.round(pts[20:, 1], 2).tolist()]]
    for name, strokes in (("browser_scale", browser), ("single_dot", [[[120], [80]]]),
                          ("horizontal_line", [[[10, 400], [50, 50]]]),
                          ("same_shape_scaled", [[[x * 2 + 7 for x in s[0]], [y * 2 + 3 for y in s[1]]] for s in samples[0]["d"]])):
        cases.append({"name": name, "strokes": strokes, "pixels": rasterize(strokes).ravel().tolist()})
    return {"spec": "shared/RASTERIZER.md", "size": 28, "tolerance": 1, "cases": cases}


def audio_fixture() -> dict:
    d = ROOT / "data" / "samples" / "speech"
    samples = json.loads((d / "samples.json").read_text())
    cases = []
    for s in [samples[0], samples[5], samples[10], samples[17]]:
        wav = af.read_wav(d / "samples" / s["file"])
        lm = af.log_mel(wav)
        cases.append({"file": s["file"], "n_samples": int(len(wav)), "log_mel": np.round(lm, 5).tolist(),
                      "raw_input_head": np.round(af.raw_input(wav)[:64], 6).tolist()})
    fb = af.mel_filterbank()
    return {"spec": "shared/AUDIO_FEATURES.md", "tolerance": 1e-3, "n_frames": af.N_FRAMES,
            "window_head": np.round(af._WINDOW[:8], 8).tolist(),
            "fbank_rows": {str(m): np.round(fb[m], 6).tolist() for m in (0, 19, 39)},
            "cases": cases}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rasterizer.json").write_text(json.dumps(rasterizer_fixture(), separators=(",", ":")))
    (OUT / "audio.json").write_text(json.dumps(audio_fixture(), separators=(",", ":")))
    print("wrote", [f"{p.name} {p.stat().st_size // 1024} KB" for p in sorted(OUT.glob("*.json"))])
