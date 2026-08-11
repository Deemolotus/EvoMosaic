"""Device selection for AMD Radeon 780M (Vulkan/RADV) — no CUDA required."""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DeviceInfo:
    torch_device: torch.device
    backend: str  # vulkan | rocm | cuda | cpu
    name: str
    notes: str = ""
    vulkan_engine: bool = False


def _allow_cpu() -> bool:
    return os.environ.get("EVO_ALLOW_CPU", "0") in {"1", "true", "yes"}


def detect_device(prefer: Optional[str] = None) -> DeviceInfo:
    """Default on this project: Vulkan (AMD RADV). CUDA is never preferred."""

    prefer = (prefer or os.environ.get("EVO_DEVICE", "vulkan")).lower()
    if prefer == "auto":
        prefer = "vulkan"

    # Explicit CPU only when allowed (smoke tests / CI).
    if prefer in {"cpu"}:
        if not _allow_cpu():
            raise RuntimeError(
                "CPU backend requested but EVO_ALLOW_CPU is not set. "
                "This machine should use Vulkan on Radeon 780M. "
                "Export EVO_ALLOW_CPU=1 only for debugging."
            )
        return DeviceInfo(torch.device("cpu"), "cpu", "CPU", "explicit CPU override")

    # Never treat NVIDIA CUDA as default on this AMD box.
    if prefer in {"cuda"} and not getattr(torch.version, "hip", None):
        raise RuntimeError(
            "CUDA requested but this host has no NVIDIA GPU. Use EVO_DEVICE=vulkan."
        )

    # ROCm/HIP PyTorch (rare on H 255 / gfx1103 — official Ryzen APU matrix is gfx115x).
    if prefer in {"rocm", "hip"} and torch.cuda.is_available() and getattr(torch.version, "hip", None):
        name = torch.cuda.get_device_name(0)
        return DeviceInfo(torch.device("cuda"), "rocm", name, "PyTorch ROCm/HIP")

    # Vulkan path (required default).
    if prefer in {"vulkan", "amd", "gpu", "radeon"}:
        try:
            from evo_lm.runtime.vulkan_backend import get_engine

            eng = get_engine()
            notes = (
                f"PyTorch host tensors + Vulkan GEMM on {eng.info.name}. "
                "Linear layers run on AMD GPU via wgpu/RADV."
            )
            return DeviceInfo(
                torch.device("cpu"),  # parameter storage; compute offloaded
                "vulkan",
                eng.info.name,
                notes,
                vulkan_engine=True,
            )
        except Exception as e:
            if _allow_cpu():
                return DeviceInfo(
                    torch.device("cpu"),
                    "cpu",
                    "CPU",
                    f"Vulkan failed ({e}); fell back because EVO_ALLOW_CPU=1",
                )
            raise RuntimeError(
                "Vulkan GPU backend failed on AMD Radeon. "
                "Check mesa-vulkan-drivers, /dev/dri/renderD128 access, "
                "and VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json. "
                f"Root cause: {e}"
            ) from e

    raise RuntimeError(f"Unknown EVO_DEVICE={prefer!r}. Use vulkan|cpu|rocm.")


def probe_vulkan() -> Optional[str]:
    try:
        from evo_lm.runtime.vulkan_backend import get_engine

        eng = get_engine()
        return f"{eng.info.name} [{eng.info.backend}]"
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["vulkaninfo", "--summary"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
        )
        for line in out.splitlines():
            if "deviceName" in line or "GPU" in line:
                return line.strip()
        return "vulkaninfo-ok"
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None


def print_device_report() -> DeviceInfo:
    info = detect_device()
    print("=" * 60)
    print("EvoLM device report (AMD / no-CUDA)")
    print(f"  platform : {platform.platform()}")
    print("  cpu      : detect via /proc or lscpu externally")
    print(f"  torch    : {torch.__version__}")
    print(f"  hip      : {getattr(torch.version, 'hip', None)}")
    print(f"  cuda_tag : {getattr(torch.version, 'cuda', None)} (ignored on this host)")
    print(f"  backend  : {info.backend}")
    print(f"  device   : {info.name}")
    print(f"  vulkan   : {info.vulkan_engine}")
    if info.notes:
        print(f"  notes    : {info.notes}")
    print("=" * 60)
    return info
