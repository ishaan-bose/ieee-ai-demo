"""Device auto-detection (SPEC 0, 6.4).

`cuda` if available, else `cpu`. Never hardcode a GPU model; VRAM is read from
torch.cuda.mem_get_info. torch is imported lazily so the module also works (as
CPU) where torch is missing; on the server NVIDIA's preinstalled build is used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

GIB = 1024**3


@dataclass(frozen=True)
class DeviceInfo:
    device: str  # "cuda" or "cpu"
    gpu_name: str | None
    vram_gb: float | None  # total VRAM of the selected GPU, GiB
    vram_free_gb: float | None
    torch_version: str | None
    cuda_version: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def _import_torch():
    try:
        import torch
    except Exception:  # missing, or a broken install
        return None
    return torch


def detect_device() -> DeviceInfo:
    torch = _import_torch()
    if torch is None:
        return DeviceInfo("cpu", None, None, None, None, None)
    version = str(torch.__version__)
    try:
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    if not has_cuda:
        return DeviceInfo("cpu", None, None, None, version, None)

    # TF32 is allowed for training (SPEC 6.4); evaluation code asks for fp32 itself.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    free, total = torch.cuda.mem_get_info(idx)
    return DeviceInfo(
        device="cuda",
        gpu_name=name,
        vram_gb=round(total / GIB, 2),
        vram_free_gb=round(free / GIB, 2),
        torch_version=version,
        cuda_version=getattr(torch.version, "cuda", None),
    )


def free_vram_bytes() -> int | None:
    """Current free VRAM in bytes (for sizing chunks later), or None on CPU."""
    torch = _import_torch()
    if torch is None or not torch.cuda.is_available():
        return None
    free, _ = torch.cuda.mem_get_info()
    return int(free)
