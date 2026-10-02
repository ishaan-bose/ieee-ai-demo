"""Startup self-check (SPEC 6.4).

Prints device / GPU name / VRAM, which dataset files are present and their sizes,
and runs a one-step training smoke test. It is cheap (array headers are read via
mmap, nothing is loaded into RAM) and never raises: problems are reported, and
the server still starts so /api/health stays reachable.

Run it on its own with:  python -m app.selfcheck
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.config import Settings
from app.data.contract import LICHESS_ARRAYS, QUICKDRAW_SPLITS, SPEECH_SPLITS
from app.device import DeviceInfo

log = logging.getLogger("selfcheck")


@dataclass
class SelfCheckResult:
    ok: bool
    device: dict
    datasets: dict = field(default_factory=dict)
    smoke_test: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "device": self.device, "datasets": self.datasets,
                "smoke_test": self.smoke_test, "warnings": self.warnings}


def _mb(path: Path) -> float:
    return round(path.stat().st_size / 1e6, 2)


def _npy(path: Path) -> dict:
    a = np.load(path, mmap_mode="r")  # header only; nothing is read into RAM
    return {"shape": list(a.shape), "dtype": str(a.dtype), "mb": _mb(path)}


def _json_len(path: Path) -> dict:
    return {"entries": len(json.loads(path.read_text())), "mb": _mb(path)}


def _dataset_summary(data_dir: Path) -> tuple[dict, list[str]]:
    out: dict = {}
    warnings: list[str] = []

    qd = data_dir / "quickdraw" / "processed"
    qd_info: dict = {}
    for split in QUICKDRAW_SPLITS:
        f = qd / f"{split}.jsonl"
        qd_info[f"{split}.jsonl"] = {"mb": _mb(f)} if f.is_file() else None
    for name in ("classes", "duel", "probe", "samples"):
        f = qd / f"{name}.json"
        qd_info[f"{name}.json"] = _json_len(f) if f.is_file() else None
    tensors = data_dir / "quickdraw" / "tensors"
    qd_info["tensors"] = (
        {f.name: _npy(f) for f in sorted(tensors.glob("*.npy"))} if tensors.is_dir()
        else "not built yet (phase 2: scripts/rasterize_quickdraw.py)"
    )
    out["quickdraw"] = qd_info

    sp = data_dir / "speech" / "processed"
    sp_info: dict = {}
    for split in SPEECH_SPLITS:
        for prefix in ("X", "y"):
            f = sp / f"{prefix}_{split}.npy"
            sp_info[f.name] = _npy(f) if f.is_file() else None
    f = sp / "samples.json"
    sp_info["samples.json"] = _json_len(f) if f.is_file() else None
    out["speech"] = sp_info

    li = data_dir / "lichess"
    out["lichess"] = {f"{name}.npy": (_npy(li / f"{name}.npy") if (li / f"{name}.npy").is_file() else None)
                      for name in LICHESS_ARRAYS}

    for ds, files in out.items():
        missing = [name for name, v in files.items() if v is None]
        if missing:
            warnings.append(f"{ds}: missing {', '.join(missing)} under {data_dir}")
    return out, warnings


def smoke_test(device: str) -> dict:
    """One SGD step on a tiny MLP. Works on CPU and GPU; falls back to tinier sizes on OOM."""
    try:
        import torch
    except Exception as e:
        return {"ok": False, "error": f"torch is not importable: {e}"}
    for batch, width in ((256, 256), (16, 16)):
        try:
            torch.manual_seed(0)
            model = torch.nn.Sequential(torch.nn.Linear(64, width), torch.nn.ReLU(), torch.nn.Linear(width, 10)).to(device)
            opt = torch.optim.SGD(model.parameters(), lr=0.1)
            x = torch.randn(batch, 64, device=device)
            y = torch.randint(0, 10, (batch,), device=device)
            t0 = time.perf_counter()
            loss = torch.nn.functional.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                loss_after = torch.nn.functional.cross_entropy(model(x), y)
            if device == "cuda":
                torch.cuda.synchronize()
            ok = bool(torch.isfinite(loss).item() and torch.isfinite(loss_after).item())
            return {"ok": ok, "device": device, "batch": batch, "width": width,
                    "loss_before": round(loss.item(), 4), "loss_after": round(loss_after.item(), 4),
                    "ms": round((time.perf_counter() - t0) * 1000, 1)}
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            return {"ok": False, "device": device, "error": f"{type(e).__name__}: {e}"}
    return {"ok": False, "device": device, "error": "out of memory even at the smallest size"}


def run_self_check(settings: Settings, device: DeviceInfo, print_fn=print) -> SelfCheckResult:
    warnings: list[str] = []
    try:
        datasets, ds_warnings = _dataset_summary(settings.data_dir)
        warnings += ds_warnings
    except Exception as e:  # a corrupt file must not stop the server
        datasets, warnings = {}, [f"dataset scan failed: {type(e).__name__}: {e}"]
    smoke = smoke_test(device.device)
    result = SelfCheckResult(ok=bool(smoke.get("ok")), device=device.to_dict(), datasets=datasets,
                             smoke_test=smoke, warnings=warnings)

    p = print_fn
    p("=" * 60)
    p("SELF-CHECK")
    p(f"  device      : {device.device}" + (f" ({device.gpu_name})" if device.gpu_name else ""))
    if device.vram_gb is not None:
        p(f"  VRAM        : {device.vram_gb} GiB total, {device.vram_free_gb} GiB free")
    p(f"  torch       : {device.torch_version or 'NOT INSTALLED'}" + (f", CUDA {device.cuda_version}" if device.cuda_version else ""))
    p(f"  DATA_DIR    : {settings.data_dir}")
    p(f"  STATE_DIR   : {settings.state_dir}")
    p(f"  NUM_WORKERS : {settings.num_workers}")
    for ds, files in datasets.items():
        listed = {k: v for k, v in files.items() if k != "tensors"}
        present = {k: v for k, v in listed.items() if v is not None}
        p(f"  {ds:<11} : {len(present)}/{len(listed)} files present")
        tensors = files.get("tensors")
        if isinstance(tensors, str):
            p(f"      {'tensors/':<16} {tensors}")
        elif isinstance(tensors, dict):
            present.update({f"tensors/{k}": v for k, v in tensors.items()})
        for name, info in present.items():
            if "shape" in info:
                p(f"      {name:<16} {info['dtype']:<6} {str(tuple(info['shape'])):<16} {info['mb']} MB")
            elif "entries" in info:
                p(f"      {name:<16} {info['entries']} entries, {info['mb']} MB")
            elif "mb" in info:
                p(f"      {name:<16} {info['mb']} MB")
    p(f"  smoke test  : {'PASS' if smoke.get('ok') else 'FAIL'} {smoke}")
    for w in warnings:
        p(f"  WARNING     : {w}")
    p(f"  result      : {'OK' if result.ok else 'PROBLEMS (server keeps running)'}")
    p("=" * 60)
    return result


if __name__ == "__main__":
    from app.config import get_settings
    from app.device import detect_device

    r = run_self_check(get_settings(), detect_device())
    raise SystemExit(0 if r.ok else 1)
