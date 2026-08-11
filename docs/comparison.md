# Hybrid vs Transformer — what the comparison means

[中文](comparison.zh.md)

Same protocol snapshot (modern_v3 dialogues, assistant-only CE, 8000 steps, Radeon 780M / ROCm):

| | Evolved hybrid | Transformer baseline |
|--|--|--|
| Structure | `moe → gru → local_attn → moe` | 7 × 256 causal Transformer |
| Params | 7.05M | 6.92M |
| Val NLL ↓ | **3.36** | 3.87 |
| Keyword coverage ↑ | **0.65** | 0.21 |
| Scramble exact match ↑ | 0.00 | 0.00 |
| Wall time | ~18 min | ~29 min |

Checkpoints: `runs/train_modern_moe_attn_v1/best.pt`, `runs/train_transformer_baseline_v1/best.pt`.

## Strengths / weaknesses

**Evolved hybrid**

- (+) Outer-loop search can keep MoE + local attention under a small budget.
- (+) In this run: better NLL, better keyword binding, less wall time.
- (−) Fitness design dominates outcomes; reorder ability did not appear.
- (−) Chat samples remain unstable (small char LM).

**Transformer baseline**

- (+) Simple, standard inductive bias; easy to explain in a tutorial.
- (+) Full causal attention is a strong binder *in principle*.
- (−) At ~7M / 8k steps here: worse NLL and keyword coverage, slower.
- (−) Still failed scramble exact match — attention alone was not enough.

## Educational takeaway

1. The **pipeline** “evolve structure → train weights” works on a laptop APU.
2. Under this budget, a searched hybrid need not lose to a small GPT stack.
3. Evolution does **not** invent skills absent from the fitness function.
4. Industry already uses evolutionary / population NAS for architecture search; this repo is a teachable miniature, not a claim of novelty over AutoML literature.

## What this is *not*

- Not a proof that “evolution beats Transformers” in general.
- Not an apples-to-apples match of identical ops (local window ≠ dense attention).
- Not a finished assistant — compositional reorder remains an open task.
