"""Decoder-only Transformer LM baseline (~7M params) for comparison with evo hybrids."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """GPT-style character LM; intentionally separate from the evolved hybrid stack."""

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 7,
        n_heads: int = 8,
        max_seq_len: int = 160,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        tie_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.tie_embeddings = tie_embeddings

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        if not tie_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

        self.moe_aux_loss = torch.zeros(())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _b, t = idx.shape
        if t > self.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max_seq_len={self.max_seq_len}")
        pos = torch.arange(t, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        # Keep attribute for drop-in parity with HybridLM train loop.
        self.moe_aux_loss = logits.new_zeros(())
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                targets[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.9,
        top_k: int | None = 40,
        eos_token_id: int | None = 3,
    ) -> torch.Tensor:
        self.eval()
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            context = idx[:, -self.max_seq_len :]
            logits, _ = self(context)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            if eos_token_id is not None:
                eos = torch.full_like(next_id, eos_token_id)
                next_id = torch.where(finished[:, None], eos, next_id)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_token_id is not None:
                finished |= next_id.squeeze(1).eq(eos_token_id)
                if bool(finished.all()):
                    break
        return idx

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config_dict(self) -> dict:
        return {
            "arch": "transformer_lm",
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "max_seq_len": self.max_seq_len,
            "tie_embeddings": self.tie_embeddings,
            "n_params": self.count_params(),
        }


def build_transformer_lm(cfg: dict, vocab_size: int) -> TransformerLM:
    model_cfg = cfg.get("model", cfg)
    return TransformerLM(
        vocab_size=vocab_size,
        d_model=int(model_cfg.get("d_model", 256)),
        n_layers=int(model_cfg.get("n_layers", 7)),
        n_heads=int(model_cfg.get("n_heads", 8)),
        max_seq_len=int(model_cfg.get("max_seq_len", 160)),
        mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        tie_embeddings=bool(model_cfg.get("tie_embeddings", True)),
    )
