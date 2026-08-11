# Transformer baseline (same budget)

[中文](transformer-baseline.zh.md)

Code lives in `baselines/transformer/` and is **outside** the evolutionary loop. It is a controlled comparison arm: same data, packing, assistant-only loss, ~7M params, 8000 steps.

## Logic in one paragraph

A decoder-only Transformer predicts the next token. Each layer does:

1. **Causal self-attention** — every position attends only to itself and the past (`is_causal=True`).
2. **FFN** — a small MLP mixing features per position.

Stack \(L\) layers, add token + position embeddings, project to vocab logits. Training uses cross-entropy; positions belonging to the user prefix are masked with `ignore_index=-100` so the main loss is on assistant answers (same protocol as the hybrid).

## Why full attention vs local attention

| | Evolved hybrid | This baseline |
|--|--|--|
| Attention | Sliding-window `local_attn` (plus SSM/GRU/MoE) | Full causal attention every layer |
| Search | Outer evolutionary loop | Fixed GPT-style stack |
| Role | Primary research object | Fair-ish reference under a param budget |

Full attention is a strong, well-understood inductive bias for binding prompt tokens to answers. It is **not** asserted to be optimal for every embedded / APU setting.

## Config

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
python -m baselines.transformer.train \
  --config baselines/transformer/configs/train_7m_modern.yaml
```

Typical size: `d_model=256`, `n_layers=7`, `n_heads=8` → ~6.92M params (tied embeddings).

Artifacts: `runs/train_transformer_baseline_v1/`.
