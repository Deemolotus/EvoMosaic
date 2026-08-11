# EvoMosaic — Evolutionary Hybrid Language Models on Consumer Hardware

[中文说明](README_zh.md)

A small **evolutionary hybrid language model** for modern Chinese dialogue. Structure is searched with a genetic outer loop; weights are trained with AdamW. Runs on **AMD Radeon 780M** via **ROCm/HIP**. A same-budget Transformer baseline lives under `baselines/transformer/` for teaching comparisons.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# ROCm torch: see scripts/setup_rocm_torch.sh
export HSA_OVERRIDE_GFX_VERSION=11.0.0   # gfx1103 / 780M
```

Modern mix (if you need to rebuild):

```bash
python -m evo_lm.data.modern
```

Evolve then train the hybrid; optionally train the Transformer baseline:

```bash
bash scripts/evolve_then_train_modern.sh
bash scripts/train_transformer_baseline.sh
```

Chat:

```bash
python -m evo_lm chat \
  --ckpt runs/train_modern_moe_attn_v1/best.pt \
  --tokenizer data/processed/tokenizer_modern_v3.json \
  --prompt $'用户: 你好\n助手:'
```

Tests: `pytest`

## Layout

```text
evo/
├── src/evo_lm/                 # evolve + hybrid LM + modern data
├── baselines/transformer/      # fixed GPT-style ~7M comparison arm
├── configs/                    # evolve / train YAMLs (final path only)
├── docs/                       # bilingual tutorials
├── scripts/                    # ROCm setup, evolve-then-train
├── data/processed/             # modern_v3 train/val + tokenizer
├── runs/                       # final hybrid + Transformer best.pt
└── tests/
```

Outer loop: population of `Genome` stacks (SSM / GRU / local attention / conv / MoE / MLP) with a hard rule of ≥1 `local_attn` and ≥1 `moe`. Fitness = assistant-answer val NLL + composition metrics − size. Inner loop: dialogue packing, loss only on tokens after `助手:`.

## Sample results

Same protocol (~7M params, 8000 steps, modern_v3, assistant-only loss, ROCm):

| Model | Params | Val NLL ↓ | Keyword cov. ↑ | Scramble EM ↑ | Wall time |
|-------|-------:|----------:|---------------:|--------------:|----------:|
| Evolved hybrid | 7.05M | **3.36** | **0.65** | 0.00 | ~18 min |
| Transformer baseline | 6.92M | 3.87 | 0.21 | 0.00 | ~29 min |

Hybrid winner structure: `moe → gru → local_attn → moe`.

## Docs

- [evolution-tutorial.md](docs/evolution-tutorial.md) — genome, fitness, outer loop
- [transformer-baseline.md](docs/transformer-baseline.md) — causal LM reference
- [comparison.md](docs/comparison.md) — strengths, limits, educational takeaway

Chinese: `*.zh.md` next to each file.

## What this shows (and does not)

Shows: evolutionary NAS + hybrid blocks can run end-to-end on a laptop APU and, under this budget, need not lose to a small Transformer. Does **not** show that evolution universally beats Transformers, or that fitness-free skills (e.g. perfect reorder) appear automatically. See [comparison.md](docs/comparison.md).

## License

MIT — see [LICENSE](LICENSE).

Data: HC3-Chinese (CC BY-SA 4.0), BAAI COIG (Apache 2.0), Complete HSK Vocabulary (MIT / CEDICT-derived). Follow `data/processed/modern_mix_v3.manifest.json` if redistributing the mix.
