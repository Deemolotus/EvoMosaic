"""Hybrid building blocks: SSM / GRU / local attention / conv / MLP / MoE.

Transformer global attention is intentionally *not* the default path.
It only appears as an optional sparse local window block that evolution
may or may not keep.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * rms


class CausalConv1d(nn.Module):
    """Depthwise causal convolution — cheap local inductive bias."""

    def __init__(self, dim: int, kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size,
            groups=dim,
            dilation=dilation,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        y = x.transpose(1, 2)
        y = F.pad(y, (self.pad, 0))
        y = self.conv(y)
        return y.transpose(1, 2)


class SelectiveSSM(nn.Module):
    """Lightweight selective state-space block (Mamba-inspired, educational).

    Uses a sequential selective scan so it runs on CPU / ROCm / Vulkan-backed
    torch without custom CUDA kernels. For production you would swap in a
    parallel scan / Vulkan compute shader.
    """

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv = CausalConv1d(self.d_inner, kernel_size=3)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_in = F.silu(self.conv(x_in))

        # [B, T, d_state + d_state + 1]
        x_dbl = self.x_proj(x_in)
        B, C, dt = torch.split(x_dbl, [self.d_state, self.d_state, 1], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # [B, T, d_inner]
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]

        y = self._selective_scan(x_in, dt, A, B, C)
        y = y + x_in * self.D
        y = y * F.silu(z)
        return self.out_proj(y)

    def _selective_scan(
        self,
        u: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Naive sequential scan — correct, portable, slow-ish (ok for <1B research)."""
        bsz, seqlen, d_inner = u.shape
        d_state = A.shape[1]
        h = u.new_zeros(bsz, d_inner, d_state)
        outs = []
        for i in range(seqlen):
            dt_i = dt[:, i].unsqueeze(-1)  # [B, D, 1]
            # discrete: h' = exp(A * dt) * h + dt * B * u
            decay = torch.exp(A.unsqueeze(0) * dt_i)  # [B, D, N]
            bu = (B[:, i].unsqueeze(1) * u[:, i].unsqueeze(-1)) * dt_i  # [B, D, N]
            h = decay * h + bu
            y_i = (h * C[:, i].unsqueeze(1)).sum(dim=-1)  # [B, D]
            outs.append(y_i)
        return torch.stack(outs, dim=1)


class GRUBlock(nn.Module):
    """Gated recurrent block with residual projection."""

    def __init__(self, d_model: int, expand: int = 1) -> None:
        super().__init__()
        hidden = d_model * expand
        self.gru = nn.GRU(d_model, hidden, batch_first=True)
        self.proj = nn.Identity() if hidden == d_model else nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(x)
        return self.proj(y)


class LocalAttention(nn.Module):
    """Sliding-window local attention (optional gene). Not global Transformer."""

    def __init__(self, d_model: int, n_heads: int = 4, window: int = 64) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window = window
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Build local causal mask
        idx = torch.arange(t, device=x.device)
        # Rows are queries and columns are keys.  A query at position i may
        # only attend to keys j where 0 <= i - j < window.
        rel = idx[:, None] - idx[None, :]
        mask = (rel >= 0) & (rel < self.window)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,T,T]

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.out(y)


class MLP(nn.Module):
    def __init__(self, d_model: int, expand: int = 4) -> None:
        super().__init__()
        hidden = d_model * expand
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


class MoE(nn.Module):
    """Tiny trainable top-1 mixture of MLP experts.

    The selected expert remains sparse. A straight-through unit gate preserves
    the previous top-1 forward behavior (and checkpoint compatibility) while
    keeping a differentiable path into the router. ``aux_loss`` is the
    Switch-style load-balancing term; callers may add it to the language-model
    objective while keeping reported NLL pure.
    """

    def __init__(self, d_model: int, n_experts: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([MLP(d_model, expand=expand) for _ in range(n_experts)])
        self.aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        idx = probs.argmax(dim=-1)  # [B, T]

        # Encourage traffic to remain balanced.  ``load`` is deliberately
        # non-differentiable; gradients flow through the mean probabilities.
        hard = F.one_hot(idx, num_classes=self.n_experts).to(probs.dtype)
        importance = probs.mean(dim=(0, 1))
        load = hard.mean(dim=(0, 1))
        self.aux_loss = self.n_experts * torch.sum(importance * load)

        out = torch.zeros_like(x)
        for e in range(self.n_experts):
            mask = idx == e
            if mask.any():
                soft_gate = probs[..., e][mask].unsqueeze(-1)
                gate = 1.0 + soft_gate - soft_gate.detach()
                out[mask] = self.experts[e](x[mask]) * gate
        return out


class HybridBlock(nn.Module):
    """One residual hybrid layer: Norm -> Op -> residual (+ optional MLP)."""

    def __init__(
        self,
        d_model: int,
        kind: str,
        *,
        d_state: int = 16,
        expand: int = 2,
        n_heads: int = 4,
        window: int = 64,
        n_experts: int = 4,
        dilation: int = 1,
        with_ffn: bool = True,
    ) -> None:
        super().__init__()
        self.kind = kind
        self.norm1 = RMSNorm(d_model)
        if kind == "ssm":
            self.op = SelectiveSSM(d_model, d_state=d_state, expand=expand)
        elif kind == "gru":
            self.op = GRUBlock(d_model, expand=1)
        elif kind == "local_attn":
            self.op = LocalAttention(d_model, n_heads=n_heads, window=window)
        elif kind == "conv":
            self.op = CausalConv1d(d_model, kernel_size=5, dilation=dilation)
        elif kind == "moe":
            self.op = MoE(d_model, n_experts=n_experts, expand=expand)
        elif kind == "mlp":
            self.op = MLP(d_model, expand=expand)
        else:
            raise ValueError(f"unknown block kind: {kind}")

        self.with_ffn = with_ffn and kind not in {"mlp", "moe"}
        if self.with_ffn:
            self.norm2 = RMSNorm(d_model)
            self.ffn = MLP(d_model, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.op(self.norm1(x))
        if self.with_ffn:
            x = x + self.ffn(self.norm2(x))
        return x
