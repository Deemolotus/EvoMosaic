"""Experimental Vulkan acceleration path for AMD GPUs.

Design goals
------------
1. Keep training on PyTorch ROCm (mature autodiff).
2. Offer a *portable* inference/math path via Vulkan compute for users who
   cannot install ROCm but have a working Mesa/AMDVLK stack.
3. Export evolved HybridLM weights to GGUF-like tensors for external
   Vulkan runtimes (llama.cpp Vulkan backend / VulkanForge-class engines).

This module ships:
- capability probe
- a tiny SPIR-V-free fallback that documents the shader contract
- host-side reference kernels matching what a Vulkan shader would compute
- an exporter that dumps state_dict + genome for Vulkan inference adapters
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from evo_lm.arch.hybrid import Genome, HybridLM


SHADER_CONTRACT = """
#version 450
// Educational contract for a Vulkan GEMM used in lm_head / projections.
// Real deployment: compile with glslangValidator -> SPIR-V, dispatch via
// VkComputePipeline. Workgroup = (16, 16, 1).

layout(local_size_x = 16, local_size_y = 16) in;
layout(binding = 0) readonly buffer A { float a[]; };
layout(binding = 1) readonly buffer B { float b[]; };
layout(binding = 2) writeonly buffer C { float c[]; };
layout(push_constant) uniform Push { uint M; uint N; uint K; } pcs;

void main() {
  uint row = gl_GlobalInvocationID.y;
  uint col = gl_GlobalInvocationID.x;
  if (row >= pcs.M || col >= pcs.N) return;
  float acc = 0.0;
  for (uint k = 0u; k < pcs.K; ++k) {
    acc += a[row * pcs.K + k] * b[k * pcs.N + col];
  }
  c[row * pcs.N + col] = acc;
}
"""


def vulkan_available() -> bool:
    try:
        import vulkan  # noqa: F401

        return True
    except Exception:
        return False


def reference_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Host reference matching the shader contract (FP32)."""
    return a @ b


def export_for_vulkan(model: HybridLM, out_dir: str | Path) -> Path:
    """Export genome + tensors for an external Vulkan inference engine."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    genome: Genome = model.genome
    genome.save(str(out / "genome.json"))
    # FP16 weights are friendlier for consumer AMD VRAM
    state = {k: v.detach().cpu().half() for k, v in model.state_dict().items()}
    torch.save(state, out / "weights.pt")
    meta = {
        "format": "evo-lm-vulkan-export-v1",
        "d_model": genome.d_model,
        "vocab_size": genome.vocab_size,
        "n_params": genome.n_params,
        "blocks": [b.kind for b in genome.blocks],
        "shader_contract": "see SHADER_CONTRACT in vulkan_ops.py",
        "suggested_runtime": [
            "llama.cpp with GGML_VULKAN=1",
            "VulkanForge-class engines for AMD RDNA",
            "custom Kompute/ash compute pipeline using SHADER_CONTRACT",
        ],
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "gemm.comp").write_text(SHADER_CONTRACT, encoding="utf-8")
    return out


def load_exported(out_dir: str | Path, device: str = "cpu") -> HybridLM:
    from evo_lm.arch.hybrid import build_model

    out = Path(out_dir)
    genome = Genome.load(str(out / "genome.json"))
    model = build_model(genome, device=device)
    state = torch.load(out / "weights.pt", map_location=device)
    model.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
    return model


def describe_amd_stack() -> dict[str, Any]:
    return {
        "training": "Install PyTorch ROCm wheels; EvoLM uses torch.device('cuda') on HIP.",
        "inference_vulkan": "Export via export_for_vulkan(); run on llama.cpp Vulkan or custom shaders.",
        "why_hybrid": "SSM/GRU/Conv path maps cleanly to sequential/scan + GEMM shaders; less KV-cache pressure than full attention.",
        "vulkan_python": vulkan_available(),
    }
