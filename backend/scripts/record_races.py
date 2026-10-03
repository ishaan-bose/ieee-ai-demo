#!/usr/bin/env python3
"""Record the race grid (SPEC 9) as cached SSE streams: frontend/public/cache/races/<key>.json (+ index.json).

Each file is ONE lane run alone on the GPU for --seconds (default 30). The frontend composes up to 3 recorded lanes into a
race, and for an uncached config replays the NEAREST cached one (`S` key). Grid: see app/training/race_grid.py
(each loss option, 7 learning rates, 6 batch sizes). Idempotent: existing recordings are kept unless --force.

    cd ~/ieee-ai-demo/backend && python3 scripts/record_races.py [--seconds 30] [--only loss lr batch] [--quick] [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO = Path(__file__).resolve().parents[2]

from app.config import get_settings  # noqa: E402
from app.data.loaders import QuickDrawData  # noqa: E402
from app.device import detect_device  # noqa: E402
from app.training.race import DoodleData, Lane, LaneConfig, run_race  # noqa: E402
from app.training.race_grid import lane_configs  # noqa: E402
from scripts._common import setup_logging, write_json  # noqa: E402


def record(key: str, kind: str, cfg: dict, data: DoodleData, seconds: float, device: str) -> dict:
    lane = Lane(LaneConfig(id=key, **cfg), data, seed=0)
    ticks: list[dict] = []
    done: dict = {}

    def emit(event: str, payload: dict) -> None:
        if event == "tick":
            ticks.append({"t": payload["t"], **{k: v for k, v in payload["lanes"][0].items() if k != "id"}})
        elif event == "done":
            done.update({"reason": payload["reason"]})

    run_race([lane], seconds, emit, lambda: False)
    return {"key": key, "kind": kind, "lane_config": cfg, "max_seconds": seconds, "device": device, "recorded_at": time.time(),
            "probe_labels": data.probe_y, "ticks": ticks, "done": done}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--only", nargs="+", choices=["loss", "lr", "batch"], default=["loss", "lr", "batch"])
    ap.add_argument("--quick", action="store_true", help="2-second recordings (rehearsal / smoke test)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--frontend-dir", type=Path, default=REPO / "frontend")
    a = ap.parse_args(argv)
    s = get_settings()
    log = setup_logging("record_races", s)
    seconds = 2.0 if a.quick else a.seconds
    dev = detect_device()
    data = DoodleData.from_dir(QuickDrawData(s.data_dir / "quickdraw"), dev.device)
    out = a.frontend_dir / "public" / "cache" / "races"
    out.mkdir(parents=True, exist_ok=True)
    done_keys, skipped = [], 0
    for kind in a.only:
        for key, cfg in lane_configs(kind):
            path = out / f"{key}.json"
            if path.is_file() and not a.force and json.loads(path.read_text()).get("max_seconds") == seconds:
                skipped += 1
                continue
            t0 = time.time()
            rec = record(key, kind, cfg, data, seconds, dev.device)
            write_json(path, rec)
            last = rec["ticks"][-1]
            log.info("%-18s %5.1fs  acc %.3f  updates %d  loss %s", key, time.time() - t0, last["acc"], last["updates"], last["loss"])
            done_keys.append(key)
    index = []
    for kind in ("loss", "lr", "batch"):
        for key, cfg in lane_configs(kind):
            if (out / f"{key}.json").is_file():
                index.append({"key": key, "kind": kind, "config": cfg})
    write_json(out / "index.json", {"version": 1, "max_seconds": seconds, "lanes": index})
    print("SUMMARY record_races:", json.dumps({"recorded": len(done_keys), "skipped_existing": skipped, "total_in_index": len(index), "seconds_each": seconds}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
