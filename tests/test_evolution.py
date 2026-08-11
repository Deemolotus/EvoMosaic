import torch

from evo_lm.arch.hybrid import BlockGene, Genome
from evo_lm.data.dataset import build_dataloaders
from evo_lm.data.tokenizer import CharTokenizer
from evo_lm.eval.fitness import corpus_fitness
from evo_lm.evolve.population import evaluate_population


def test_corpus_fitness_is_reproducible_for_same_seed():
    blocks = [f"用户: 问题{i}\n助手: 回答{i}" for i in range(8)]
    text = "\n\n".join(blocks * 2)
    tokenizer = CharTokenizer.train_from_texts([text], min_freq=1)
    train_loader, val_loader = build_dataloaders(
        text,
        tokenizer,
        seq_len=8,
        batch_size=2,
        val_ratio=0.25,
        split_seed=9,
        shuffle_train=False,
        fixed_train_order=True,
        pack_mode="dialogue",
        assistant_loss_only=True,
    )
    batch_a = next(iter(train_loader))
    batch_b = next(iter(train_loader))
    assert torch.equal(batch_a[0], batch_b[0])
    assert torch.equal(batch_a[1], batch_b[1])
    genome = Genome(
        d_model=8,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=8,
        blocks=[
            BlockGene(kind="local_attn", n_heads=2, window=8, expand=1),
            BlockGene(kind="moe", n_experts=2, expand=1),
        ],
    )

    kwargs = dict(
        device=torch.device("cpu"),
        train_steps=1,
        seed=77,
        use_vulkan=False,
    )
    first = corpus_fitness(genome, train_loader, val_loader, **kwargs)
    second = corpus_fitness(genome, train_loader, val_loader, **kwargs)
    assert first == second


def test_population_evaluation_caches_identical_genomes():
    genome = Genome(d_model=8, vocab_size=16, blocks=[BlockGene(kind="conv")])
    clone = Genome.from_dict(genome.to_dict())
    calls = 0

    def fitness_fn(_genome):
        nonlocal calls
        calls += 1
        return 1.25

    cache = {}
    evaluate_population([genome, clone], fitness_fn, cache)
    assert calls == 1
    assert genome.fitness == clone.fitness == 1.25


def test_compositional_task_metrics_helpers():
    from evo_lm.eval.fitness import (
        keyword_coverage,
        parse_compositional_tasks,
        scramble_exact_match,
    )

    blocks = [
        '用户: 请用“学习”和“汉语”组成一句自然的话。\n助手: 我们认真学习汉语。',
        "用户: 请把这些片段排列成自然句子：我｜们｜学习\n助手: 我们学习",
    ]
    composition, scramble = parse_compositional_tasks(blocks)
    assert len(composition) == 1
    assert composition[0]["keywords"] == ["学习", "汉语"]
    assert len(scramble) == 1
    assert keyword_coverage("今天一起学习汉语吧", ["学习", "汉语"]) == 1.0
    assert scramble_exact_match("我们学习", "我们学习") == 1.0
    assert scramble_exact_match("别的句子", "我们学习") == 0.0
