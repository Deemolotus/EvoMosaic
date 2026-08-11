import os
import random

# Unit tests may run without GPU assertion path
os.environ.setdefault("EVO_ALLOW_CPU", "1")
os.environ.setdefault("EVO_DEVICE", "cpu")

from evo_lm.arch.blocks import LocalAttention, MoE
from evo_lm.arch.hybrid import (
    BlockGene,
    Genome,
    HybridLM,
    build_model,
    crossover,
    enforce_arch_constraints,
    estimate_params,
    enforce_param_budget,
)


def test_genome_roundtrip(tmp_path):
    rng = random.Random(0)
    g = Genome.random(rng, d_model=128, n_blocks=6, vocab_size=1000)
    path = tmp_path / "g.json"
    g.save(str(path))
    g2 = Genome.load(str(path))
    assert g2.d_model == g.d_model
    assert len(g2.blocks) == len(g.blocks)


def test_build_and_forward():
    rng = random.Random(1)
    g = Genome.random(rng, d_model=64, n_blocks=4, vocab_size=200, max_seq_len=32)
    g = enforce_param_budget(g, 5_000_000, rng)
    model = build_model(g, device="cpu", use_vulkan=False)
    import torch

    x = torch.randint(0, g.vocab_size, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, g.vocab_size)
    assert loss is not None
    assert model.count_params() > 0


def test_crossover_and_mutate():
    rng = random.Random(2)
    a = Genome.random(rng, d_model=64, n_blocks=5, vocab_size=100)
    b = Genome.random(rng, d_model=64, n_blocks=7, vocab_size=100)
    a.fitness = -1.0
    b.fitness = -2.0
    child = crossover(a, b, rng)
    child2 = child.mutate(rng)
    assert len(child2.blocks) >= 1
    assert estimate_params(child2) > 0
    assert any(b.kind == "local_attn" for b in child2.blocks)
    assert any(b.kind == "moe" for b in child2.blocks)


def test_arch_constraints_force_attn_and_moe():
    rng = random.Random(3)
    g = Genome(
        d_model=64,
        vocab_size=100,
        blocks=[BlockGene(kind="conv"), BlockGene(kind="gru")],
    )
    g = enforce_arch_constraints(g, rng)
    kinds = [b.kind for b in g.blocks]
    assert kinds.count("local_attn") >= 1
    assert kinds.count("moe") >= 1

    for _ in range(20):
        g = g.mutate(rng)
        g = enforce_param_budget(g, 2_000_000, rng)
        kinds = [b.kind for b in g.blocks]
        assert kinds.count("local_attn") >= 1
        assert kinds.count("moe") >= 1


def test_local_attention_cannot_see_future():
    import torch

    torch.manual_seed(0)
    attention = LocalAttention(16, n_heads=4, window=8).eval()
    original = torch.randn(2, 7, 16)
    changed = original.clone()
    changed[:, -1] += 100.0

    before = attention(original)
    after = attention(changed)
    assert torch.allclose(before[:, :-1], after[:, :-1], atol=1e-6, rtol=1e-6)


def test_moe_router_receives_gradient():
    import torch

    torch.manual_seed(0)
    moe = MoE(16, n_experts=4, expand=2)
    out = moe(torch.randn(3, 5, 16))
    objective = out.square().mean() + 0.01 * moe.aux_loss
    objective.backward()

    assert moe.router.weight.grad is not None
    assert torch.isfinite(moe.router.weight.grad).all()
    assert moe.router.weight.grad.abs().sum() > 0


def test_parameter_estimate_matches_model():
    for kind in ("ssm", "gru", "local_attn", "conv", "moe", "mlp"):
        for tied in (True, False):
            genome = Genome(
                d_model=16,
                vocab_size=100,
                blocks=[BlockGene(kind=kind, d_state=8, expand=2, n_heads=4, n_experts=3)],
                tie_embeddings=tied,
            )
            assert estimate_params(genome) == HybridLM(genome).count_params()