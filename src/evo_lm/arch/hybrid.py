"""Genome ↔ phenotype: evolve hybrid stack layouts under a param budget."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from typing import Any, List

from .blocks import HybridBlock, RMSNorm
import torch
import torch.nn as nn


BLOCK_KINDS = ("ssm", "gru", "local_attn", "conv", "moe", "mlp")

# Soft prior among residual slots. Hard constraints still force ≥1 local_attn + ≥1 moe.
KIND_PRIOR = {
    "ssm": 0.28,
    "gru": 0.22,
    "conv": 0.16,
    "moe": 0.16,
    "local_attn": 0.12,
    "mlp": 0.06,
}


@dataclass
class BlockGene:
    kind: str = "ssm"
    d_state: int = 16
    expand: int = 2
    n_heads: int = 4
    window: int = 64
    n_experts: int = 4
    dilation: int = 1
    with_ffn: bool = True

    def mutate(self, rng: random.Random) -> None:
        if rng.random() < 0.25:
            self.kind = rng.choices(list(KIND_PRIOR), weights=list(KIND_PRIOR.values()), k=1)[0]
        if rng.random() < 0.2:
            self.d_state = rng.choice([8, 16, 24, 32, 48, 64])
        if rng.random() < 0.2:
            self.expand = rng.choice([1, 2, 3, 4])
        if rng.random() < 0.15:
            self.n_heads = rng.choice([2, 4, 8])
        if rng.random() < 0.15:
            self.window = rng.choice([32, 64, 128, 256])
        if rng.random() < 0.15:
            self.n_experts = rng.choice([2, 4, 8])
        if rng.random() < 0.15:
            self.dilation = rng.choice([1, 2, 4])
        if rng.random() < 0.1:
            self.with_ffn = not self.with_ffn


@dataclass
class Genome:
    """Compact encoding of a hybrid LM under ~10B params."""

    d_model: int = 256
    vocab_size: int = 8000
    max_seq_len: int = 512
    blocks: List[BlockGene] = field(default_factory=list)
    tie_embeddings: bool = True
    # bookkeeping
    generation: int = 0
    fitness: float = float("-inf")
    n_params: int = 0

    @staticmethod
    def random(
        rng: random.Random,
        *,
        d_model: int = 256,
        n_blocks: int | None = None,
        vocab_size: int = 8000,
        max_seq_len: int = 512,
    ) -> "Genome":
        n = n_blocks if n_blocks is not None else rng.randint(4, 12)
        blocks = []
        for _ in range(n):
            kind = rng.choices(list(KIND_PRIOR), weights=list(KIND_PRIOR.values()), k=1)[0]
            blocks.append(
                BlockGene(
                    kind=kind,
                    d_state=rng.choice([8, 16, 32]),
                    expand=rng.choice([1, 2, 4]),
                    n_heads=rng.choice([2, 4]),
                    window=rng.choice([32, 64, 128]),
                    n_experts=rng.choice([2, 4]),
                    dilation=rng.choice([1, 2]),
                    with_ffn=rng.random() < 0.8,
                )
            )
        g = Genome(d_model=d_model, vocab_size=vocab_size, max_seq_len=max_seq_len, blocks=blocks)
        return enforce_arch_constraints(g, rng)

    def mutate(self, rng: random.Random, max_blocks: int = 48) -> "Genome":
        child = Genome(
            d_model=self.d_model,
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
            blocks=[BlockGene(**asdict(b)) for b in self.blocks],
            tie_embeddings=self.tie_embeddings,
            generation=self.generation + 1,
        )
        # structural mutations (NEAT-flavored)
        r = rng.random()
        if r < 0.2 and len(child.blocks) < max_blocks:
            idx = rng.randint(0, len(child.blocks))
            kind = rng.choices(list(KIND_PRIOR), weights=list(KIND_PRIOR.values()), k=1)[0]
            child.blocks.insert(idx, BlockGene(kind=kind))
        elif r < 0.35 and len(child.blocks) > 2:
            deletable = [
                i
                for i, b in enumerate(child.blocks)
                if _can_delete_block(child, i)
            ]
            if deletable:
                del child.blocks[rng.choice(deletable)]
        elif r < 0.5 and len(child.blocks) >= 2:
            i, j = rng.sample(range(len(child.blocks)), 2)
            child.blocks[i], child.blocks[j] = child.blocks[j], child.blocks[i]
        else:
            if child.blocks:
                idx = rng.randrange(len(child.blocks))
                before = child.blocks[idx].kind
                child.blocks[idx].mutate(rng)
                # Keep forced kinds: if mutation dropped the last required block, revert kind.
                if before in {"local_attn", "moe"} and child.blocks[idx].kind != before:
                    if sum(1 for b in child.blocks if b.kind == before) < 1:
                        child.blocks[idx].kind = before

        # rare width mutation
        if rng.random() < 0.05:
            child.d_model = max(64, min(4096, child.d_model + rng.choice([-64, 64, 128, -128])))
            # keep heads divisible later in builder
        return enforce_arch_constraints(child, rng)

    def to_dict(self) -> dict[str, Any]:
        return {
            "d_model": self.d_model,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "tie_embeddings": self.tie_embeddings,
            "generation": self.generation,
            "fitness": self.fitness,
            "n_params": self.n_params,
            "blocks": [asdict(b) for b in self.blocks],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Genome":
        blocks = [BlockGene(**b) for b in d["blocks"]]
        g = Genome(
            d_model=d["d_model"],
            vocab_size=d["vocab_size"],
            max_seq_len=d.get("max_seq_len", 512),
            blocks=blocks,
            tie_embeddings=d.get("tie_embeddings", True),
            generation=d.get("generation", 0),
            fitness=d.get("fitness", float("-inf")),
            n_params=d.get("n_params", 0),
        )
        if not g.n_params:
            g.n_params = estimate_params(g)
        return g

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> "Genome":
        with open(path, encoding="utf-8") as f:
            return Genome.from_dict(json.load(f))


def crossover(a: Genome, b: Genome, rng: random.Random) -> Genome:
    """One-point crossover over block sequences + inherit globals from fitter parent."""
    parent = a if a.fitness >= b.fitness else b
    other = b if parent is a else a
    if not parent.blocks or not other.blocks:
        return parent.mutate(rng)
    cut_a = rng.randint(1, len(parent.blocks))
    cut_b = rng.randint(0, len(other.blocks))
    blocks = [BlockGene(**asdict(x)) for x in parent.blocks[:cut_a] + other.blocks[cut_b:]]
    child = Genome(
        d_model=parent.d_model if rng.random() < 0.7 else other.d_model,
        vocab_size=parent.vocab_size,
        max_seq_len=parent.max_seq_len,
        blocks=blocks or [BlockGene()],
        tie_embeddings=parent.tie_embeddings,
        generation=max(a.generation, b.generation) + 1,
    )
    return enforce_arch_constraints(child, rng)


def estimate_params(g: Genome) -> int:
    """Analytic parameter count matching ``HybridLM`` modules."""
    d = g.d_model
    v = g.vocab_size
    n = v * d  # embedding
    for b in g.blocks:
        if b.kind == "ssm":
            inner = d * b.expand
            n += d * inner * 2  # in_proj
            n += inner * 3 + inner  # depthwise conv weight + bias
            n += inner * (b.d_state * 2 + 1)  # x_proj
            n += inner * 2  # dt_proj weight + bias
            n += inner * b.d_state  # A_log
            n += inner  # D
            n += inner * d  # out_proj
            if b.with_ffn:
                n += d * (d * b.expand) * 2
        elif b.kind == "gru":
            # PyTorch GRU: input/recurrent matrices and two bias vectors.
            h = d
            n += 3 * d * h + 3 * h * h + 6 * h
            if b.with_ffn:
                n += d * (d * b.expand) * 2
        elif b.kind == "local_attn":
            n += 4 * d * d
            if b.with_ffn:
                n += d * (d * b.expand) * 2
        elif b.kind == "conv":
            n += d * 5 + d  # depthwise k=5 + bias
            if b.with_ffn:
                n += d * (d * b.expand) * 2
        elif b.kind == "moe":
            n += d * b.n_experts + b.n_experts * (d * (d * b.expand) * 2)
        elif b.kind == "mlp":
            n += d * (d * b.expand) * 2
        n += d  # RMSNorm
        if b.with_ffn and b.kind not in {"mlp", "moe"}:
            n += d
    n += d  # final norm
    n += 0 if g.tie_embeddings else v * d
    return int(n)


class HybridLM(nn.Module):
    def __init__(self, genome: Genome) -> None:
        super().__init__()
        self.genome = genome
        d = genome.d_model
        self.tok_emb = nn.Embedding(genome.vocab_size, d)
        # Keep d_model divisible by n_heads for local_attn genes
        layers: list[nn.Module] = []
        for gene in genome.blocks:
            heads = gene.n_heads
            while d % heads != 0 and heads > 1:
                heads //= 2
            layers.append(
                HybridBlock(
                    d,
                    gene.kind,
                    d_state=gene.d_state,
                    expand=gene.expand,
                    n_heads=max(1, heads),
                    window=gene.window,
                    n_experts=gene.n_experts,
                    dilation=gene.dilation,
                    with_ffn=gene.with_ffn,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, genome.vocab_size, bias=False)
        self.moe_aux_loss: torch.Tensor | None = None
        if genome.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        # PyTorch Embedding defaults to std=1, which makes tied next-token
        # logits extremely overconfident and lets short-fitness rankings be
        # dominated by initialization scale rather than architecture quality.
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        if not genome.tie_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.tok_emb(idx)
        moe_aux_losses = []
        for layer in self.layers:
            x = layer(x)
            aux_loss = getattr(layer.op, "aux_loss", None)
            if aux_loss is not None:
                moe_aux_losses.append(aux_loss)
        x = self.norm(x)
        logits = self.lm_head(x)
        self.moe_aux_loss = (
            torch.stack(moe_aux_losses).mean()
            if moe_aux_losses
            else logits.new_zeros(())
        )
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
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
            context = idx[:, -self.genome.max_seq_len :]
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


def build_model(
    genome: Genome,
    device: torch.device | str = "cpu",
    *,
    use_vulkan: bool | None = None,
) -> HybridLM:
    """Build phenotype. When ``use_vulkan`` (default: EVO_DEVICE=vulkan), Linear→VulkanLinear."""
    import os

    model = HybridLM(genome)
    genome.n_params = model.count_params()
    model = model.to(device)

    if use_vulkan is None:
        use_vulkan = os.environ.get("EVO_DEVICE", "vulkan").lower() in {
            "vulkan",
            "amd",
            "gpu",
            "radeon",
            "auto",
        }
    if use_vulkan:
        from evo_lm.runtime.vulkan_backend import replace_linears_with_vulkan

        n = replace_linears_with_vulkan(model)
        if genome.tie_embeddings:
            # re-bind after Linear→VulkanLinear swap
            model.lm_head.weight = model.tok_emb.weight
        print(f"[vulkan] replaced {n} Linear layers with VulkanLinear (AMD RADV GEMM)")
    return model


def _kind_count(genome: Genome, kind: str) -> int:
    return sum(1 for b in genome.blocks if b.kind == kind)


def _can_delete_block(
    genome: Genome,
    idx: int,
    *,
    min_local_attn: int = 1,
    min_moe: int = 1,
) -> bool:
    kind = genome.blocks[idx].kind
    if kind == "local_attn" and _kind_count(genome, "local_attn") <= min_local_attn:
        return False
    if kind == "moe" and _kind_count(genome, "moe") <= min_moe:
        return False
    return True


def enforce_arch_constraints(
    genome: Genome,
    rng: random.Random,
    *,
    min_local_attn: int = 1,
    min_moe: int = 1,
) -> Genome:
    """Insert missing required blocks: at least one local attention and one MoE."""
    g = genome
    while _kind_count(g, "local_attn") < min_local_attn:
        idx = rng.randint(0, len(g.blocks))
        g.blocks.insert(
            idx,
            BlockGene(kind="local_attn", n_heads=4, window=64, expand=2, with_ffn=True),
        )
    while _kind_count(g, "moe") < min_moe:
        idx = rng.randint(0, len(g.blocks))
        g.blocks.insert(
            idx,
            BlockGene(kind="moe", n_experts=4, expand=2, with_ffn=False),
        )
    if not g.blocks:
        g.blocks = [
            BlockGene(kind="local_attn", n_heads=4, window=64, expand=2),
            BlockGene(kind="moe", n_experts=4, expand=2),
        ]
    g.n_params = estimate_params(g)
    return g


def enforce_param_budget(
    genome: Genome,
    max_params: int,
    rng: random.Random,
    *,
    min_local_attn: int = 1,
    min_moe: int = 1,
) -> Genome:
    """Shrink genome until estimated params fit under budget (<10B etc.)."""
    g = enforce_arch_constraints(
        genome,
        rng,
        min_local_attn=min_local_attn,
        min_moe=min_moe,
    )
    guard = 0
    while estimate_params(g) > max_params and guard < 200:
        deletable = [
            i
            for i in range(len(g.blocks))
            if _can_delete_block(
                g,
                i,
                min_local_attn=min_local_attn,
                min_moe=min_moe,
            )
        ]
        if len(deletable) > 0 and len(g.blocks) > min_local_attn + min_moe and rng.random() < 0.6:
            del g.blocks[rng.choice(deletable)]
        elif g.d_model > 128:
            g.d_model = max(128, g.d_model - 64)
        else:
            # shrink expand factors on non-protected or any block
            b = rng.choice(g.blocks)
            b.expand = max(1, b.expand - 1)
            if b.kind == "moe" and b.n_experts > 2 and rng.random() < 0.5:
                b.n_experts = max(2, b.n_experts // 2)
        g = enforce_arch_constraints(
            g,
            rng,
            min_local_attn=min_local_attn,
            min_moe=min_moe,
        )
        guard += 1
    if estimate_params(g) > max_params:
        raise ValueError(
            f"Could not fit genome into max_params={max_params:,}; "
            f"smallest attempted estimate is {estimate_params(g):,}"
        )
    return g
