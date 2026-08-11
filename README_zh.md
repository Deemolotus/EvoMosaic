# EvoMosaic — Evolutionary Hybrid Language Models on Consumer Hardware

[English](README.md)

面向现代中文对话的小型**进化混合语言模型**：外环用遗传算法搜结构，内环用 AdamW 训权重。在 **AMD Radeon 780M** 上走 **ROCm/HIP**。同预算 Transformer 对照在 `baselines/transformer/`，方便教学对比。

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# ROCm 版 torch：见 scripts/setup_rocm_torch.sh
export HSA_OVERRIDE_GFX_VERSION=11.0.0   # gfx1103 / 780M
```

如需重建现代混料：

```bash
python -m evo_lm.data.modern
```

进化并训练混合体；可选再训 Transformer 基线：

```bash
bash scripts/evolve_then_train_modern.sh
bash scripts/train_transformer_baseline.sh
```

闲聊：

```bash
python -m evo_lm chat \
  --ckpt runs/train_modern_moe_attn_v1/best.pt \
  --tokenizer data/processed/tokenizer_modern_v3.json \
  --prompt $'用户: 你好\n助手:'
```

测试：`pytest`

## 目录

```text
evo/
├── src/evo_lm/                 # 进化 + 混合 LM + 现代语料
├── baselines/transformer/      # 固定 GPT 风格 ~7M 对照
├── configs/                    # 最终 evolve / train 配置
├── docs/                       # 双语教程
├── scripts/                    # ROCm、进化后训练
├── data/processed/             # modern_v3 训练/验证 + 词表
├── runs/                       # 最终混合体与 Transformer 的 best.pt
└── tests/
```

外环：`Genome` 块序列（SSM / GRU / 局部注意力 / 卷积 / MoE / MLP），硬约束 ≥1 个 `local_attn` 与 ≥1 个 `moe`。适应度 = 助手答案验证 NLL + 组句指标 − 尺寸。内环：对话打包，损失只算 `助手:` 之后。

## 数据与运行结果

由于文件体积较大，`data/` 和 `runs/` 目录未直接包含在本仓库中。

你可以通过 [Google Drive](https://drive.google.com/file/d/11hBatLUrNLL3Mzn_fXTaJ7zoB8nsTCpq/view?usp=sharing) 下载包含这两个目录的 ZIP 压缩包。

下载完成后，请将压缩包解压到项目根目录。

## 部分结果

同一协议（约 7M、8000 步、modern_v3、助手答案损失、ROCm）：

| 模型 | 参数 | 验证 NLL ↓ | 关键词覆盖 ↑ | 乱序精确率 ↑ | 墙钟 |
|------|-----:|-----------:|-------------:|-------------:|-----:|
| 进化混合体 | 7.05M | **3.36** | **0.65** | 0.00 | ~18 min |
| Transformer 基线 | 6.92M | 3.87 | 0.21 | 0.00 | ~29 min |

混合体优胜结构：`moe → gru → local_attn → moe`。

## 技术背景

EvoMosaic 结合了多种已有神经网络架构与训练方法的思想。

项目的进化外循环参考了进化神经架构搜索（Real et al., 2019）。混合架构搜索空间包含 GRU 模块（Cho et al., 2014）、Transformer 风格的局部注意力模块（Vaswani et al., 2017）、混合专家（Mixture-of-Experts）模块（Shazeer et al., 2017），以及受状态空间模型启发的序列模块（Gu et al., 2022）。模型权重使用 AdamW 进行优化（Loshchilov & Hutter, 2019）。

中文对话数据混合包含来自 HC3-Chinese（Guo et al., 2023）、COIG（Zhang et al., 2023）以及 Complete HSK Vocabulary（Zafirópulos, n.d.）的数据。

## 文档

- [evolution-tutorial.zh.md](docs/evolution-tutorial.zh.md) — 基因组、适应度、外环
- [transformer-baseline.zh.md](docs/transformer-baseline.zh.md) — 因果 LM 对照
- [comparison.zh.md](docs/comparison.zh.md) — 长短、边界、教育意义

英文版是同目录去掉 `.zh` 的文件。

## 能说明什么，不能说明什么

能说明：进化 NAS + 混合块在本机 APU 上可端到端跑通；同预算下不必输给小 Transformer。**不能**说明进化普遍碾压 Transformer，也不能说明适应度没写的技能（如完美乱序）会自动出现。详见 [comparison.zh.md](docs/comparison.zh.md)。

## 参考文献

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

MIT，见 [LICENSE](LICENSE)。

数据：HC3-Chinese（CC BY-SA 4.0）、BAAI COIG（Apache 2.0）、Complete HSK Vocabulary（MIT / CEDICT 衍生）。再分发混料时遵守 `data/processed/modern_mix_v3.manifest.json`。
