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
EvoMosaic/
├── src/evo_lm/                 # Core implementation: evolution, hybrid LM, training, and chat
├── baselines/transformer/      # ~7M parameter Transformer baseline model
├── configs/                    # Architecture-search and training configurations
├── scripts/                    # ROCm setup, evolution, training, and utility scripts
├── docs/                       # Bilingual tutorials and project documentation
├── tests/                      # Tests
├── data/processed/             # Processed datasets and tokenizer (download separately)
└── runs/                       # Checkpoints and experiment results (download separately)
```

Outer loop: population of `Genome` stacks (SSM / GRU / local attention / conv / MoE / MLP) with a hard rule of ≥1 `local_attn` and ≥1 `moe`. Fitness = assistant-answer val NLL + composition metrics − size. Inner loop: dialogue packing, loss only on tokens after `助手:`.

## Data and Run Results

The `data/` and `runs/` directories are not included directly in this repository due to their size.

You can download both directories as a ZIP archive from [Google Drive](https://drive.google.com/file/d/11hBatLUrNLL3Mzn_fXTaJ7zoB8nsTCpq/view?usp=sharing).

After downloading, extract the archive into the project root.

## Sample results

Same protocol (~7M params, 8000 steps, modern_v3, assistant-only loss, ROCm):

| Model | Params | Val NLL ↓ | Keyword cov. ↑ | Scramble EM ↑ | Wall time |
|-------|-------:|----------:|---------------:|--------------:|----------:|
| Evolved hybrid | 7.05M | **3.36** | **0.65** | 0.00 | ~18 min |
| Transformer baseline | 6.92M | 3.87 | 0.21 | 0.00 | ~29 min |

Hybrid winner structure: `moe → gru → local_attn → moe`.

## Technical Background

EvoMosaic combines ideas from several established neural network architectures and training methods.

The evolutionary outer loop is inspired by evolutionary neural architecture search (Real et al., 2019). The hybrid search space includes GRU blocks (Cho et al., 2014), Transformer-style local attention (Vaswani et al., 2017), Mixture-of-Experts blocks (Shazeer et al., 2017), and state-space-model-inspired sequence blocks (Gu et al., 2022). Model weights are optimized with AdamW (Loshchilov & Hutter, 2019).

The Chinese dialogue data mix includes data derived from HC3-Chinese (Guo et al., 2023), COIG (Zhang et al., 2023), and Complete HSK Vocabulary (Zafirópulos, n.d.).

## Docs

- [evolution-tutorial.md](docs/evolution-tutorial.md) — genome, fitness, outer loop
- [transformer-baseline.md](docs/transformer-baseline.md) — causal LM reference
- [comparison.md](docs/comparison.md) — strengths, limits, educational takeaway

Chinese: `*.zh.md` next to each file.

## What this shows (and does not)

Shows: evolutionary NAS + hybrid blocks can run end-to-end on a laptop APU and, under this budget, need not lose to a small Transformer. Does **not** show that evolution universally beats Transformers, or that fitness-free skills (e.g. perfect reorder) appear automatically. See [comparison.md](docs/comparison.md).

## References

Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 1724–1734. https://doi.org/10.3115/v1/D14-1179

Gu, A., Goel, K., & Ré, C. (2022). Efficiently modeling long sequences with structured state spaces. *International Conference on Learning Representations (ICLR).* https://openreview.net/forum?id=uYLFoz1vlAC

Guo, B., Zhang, X., Wang, Z., Jiang, M., Nie, J., Ding, Y., Yue, J., & Wu, Y. (2023). How close is ChatGPT to human experts? Comparison corpus, evaluation, and detection. *arXiv*. https://doi.org/10.48550/arXiv.2301.07597

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *International Conference on Learning Representations (ICLR).* https://openreview.net/forum?id=Bkg6RiCqY7

Real, E., Aggarwal, A., Huang, Y., & Le, Q. V. (2019). Regularized evolution for image classifier architecture search. *Proceedings of the AAAI Conference on Artificial Intelligence, 33*(01), 4780–4789. https://doi.org/10.1609/aaai.v33i01.33014780

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q. V., Hinton, G. E., & Dean, J. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *International Conference on Learning Representations (ICLR).* https://openreview.net/forum?id=B1ckMDqlg

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems, 30*, 5998–6008. https://arxiv.org/abs/1706.03762

Zafirópulos, Y. (n.d.). *Complete HSK Vocabulary* [Data set]. GitHub. https://github.com/drkameleon/complete-hsk-vocabulary

Zhang, G., Shi, Y., Liu, R., Yuan, R., Li, Y., Dong, S., Shu, Y., Li, Z., Wang, Z., Lin, C., Huang, W., & Fu, J. (2023). Chinese open instruction generalist: A preliminary release. *arXiv*. https://doi.org/10.48550/arXiv.2304.07987

## License

MIT — see [LICENSE](LICENSE).

Data: HC3-Chinese (CC BY-SA 4.0), BAAI COIG (Apache 2.0), Complete HSK Vocabulary (MIT / CEDICT-derived). Follow `data/processed/modern_mix_v3.manifest.json` if redistributing the mix.
