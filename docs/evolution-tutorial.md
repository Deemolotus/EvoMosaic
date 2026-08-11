# How evolutionary architecture search works

[中文](evolution-tutorial.zh.md)

This project uses **evolutionary NAS**: the outer loop searches *structure* (which blocks to stack), the inner loop trains *weights* with gradient descent.

## Genome → phenotype

A candidate is a `Genome` (`src/evo_lm/arch/hybrid.py`):

```text
Genome
├── d_model, vocab_size, max_seq_len
└── blocks[]  (sequence of BlockGene)
      kind ∈ {ssm, gru, local_attn, conv, moe, mlp}
      expand, d_state, n_heads, window, n_experts, …
```

Building `HybridLM` from a genome is the phenotype. Soft sampling prior prefers SSM/GRU/conv; **hard constraints** still require:

- at least one `local_attn`
- at least one `moe`

## Outer-loop cycle

```text
init population  →  evaluate fitness  →  keep elites
       ↑                    │
       │                    ↓
       └── mutate / crossover parents (tournament)
```

Implemented in `src/evo_lm/evolve/population.py` → `run_evolution`.

Each fitness call (`corpus` mode in `src/evo_lm/eval/fitness.py`):

1. Build the model on device.
2. Short AdamW burst on modern dialogue batches (assistant-answer loss only).
3. Measure validation NLL on answer tokens.
4. Generate a few composition / scramble prompts; score keyword coverage and exact match.
5. Return roughly:

```text
fitness = -val_NLL
        + λ_kw * keyword_coverage
        + λ_sc * scramble_exact_match
        - size_penalty
```

Higher is better. Genomes over the param budget are rejected.

## Worked sketch of fitness

Suppose after the short train:

- val NLL = 6.4  
- keyword coverage = 0.25  
- scramble EM = 0.0  
- params = 7e6, size penalty ≈ 0.00035  

With `λ_kw=0.35`, `λ_sc=0.45`:

```text
fitness ≈ -6.4 + 0.35*0.25 + 0.45*0.0 - 0.00035 ≈ -6.31
```

So lowering NLL and raising compositional scores both move the population.

## Commands

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0   # AMD Radeon 780M / gfx1103
python -m evo_lm evolve --config configs/evolve_modern_moe_attn_v1.yaml
```

Winner: `runs/train_modern_moe_attn_v1/best_genome.json` (copied next to the trained weights).

```bash
python -m evo_lm train --config configs/train_modern_moe_attn_v1.yaml
```

## What evolution does *not* do

It does **not** discover algorithms you never put in the fitness. Scramble exact-match stayed at 0 in our runs — the search rewarded models that lowered LM loss and often hit keywords, not perfect reorder solvers.
