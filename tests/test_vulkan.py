"""Vulkan GEMM smoke test — requires AMD RADV (/dev/dri)."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/radeon_icd.json")
os.environ.setdefault("WGPU_BACKEND_TYPE", "Vulkan")
os.environ.setdefault("EVO_DEVICE", "vulkan")
os.environ.setdefault("EVO_VULKAN_FORCE_SMALL", "1")


@pytest.fixture(scope="module")
def engine():
    try:
        from evo_lm.runtime.vulkan_backend import get_engine

        return get_engine()
    except Exception as e:
        pytest.skip(f"Vulkan not available: {e}")


def test_radeon_adapter(engine):
    assert engine.info.backend.lower() == "vulkan"
    assert "radeon" in engine.info.name.lower() or "amd" in engine.info.name.lower()


def test_vulkan_gemm_matches_numpy(engine):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((64, 48), dtype=np.float32)
    b = rng.standard_normal((48, 32), dtype=np.float32)
    before = engine.gemm_calls
    got = engine.gemm(a, b)
    ref = a @ b
    assert got.shape == ref.shape
    assert np.allclose(got, ref, rtol=1e-3, atol=1e-3)
    assert engine.gemm_calls > before


def test_vulkan_linear_backward(engine):
    from evo_lm.runtime.vulkan_backend import VulkanLinear

    layer = VulkanLinear(32, 16)
    x = torch.randn(4, 8, 32, requires_grad=True)
    y = layer(x)
    y.sum().backward()
    assert x.grad is not None
    assert layer.weight.grad is not None
    assert engine.gemm_calls >= 1


def test_detect_device_vulkan(engine, monkeypatch):
    from evo_lm.runtime.device import detect_device

    monkeypatch.setenv("EVO_DEVICE", "vulkan")
    monkeypatch.delenv("EVO_ALLOW_CPU", raising=False)
    info = detect_device("vulkan")
    assert info.backend == "vulkan"
    assert info.vulkan_engine
