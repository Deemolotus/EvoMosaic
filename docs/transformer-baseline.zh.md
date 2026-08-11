# Transformer 基线（同预算对照）

[English](transformer-baseline.md)

代码在 `baselines/transformer/`，**不在**进化循环里。它是对照臂：同一数据、同一打包与助手答案损失、约 7M 参数、8000 步。

## 一句话逻辑

Decoder-only Transformer 做下一词预测。每一层：

1. **因果自注意力** — 每个位置只能看见自己与过去（`is_causal=True`）；
2. **FFN** — 按位置的小型 MLP。

堆 \(L\) 层，加上 token / 位置嵌入，投影到词表 logits。训练用交叉熵；用户前缀位置用 `ignore_index=-100` 掩掉，主损失只落在助手答案上（与混合体同一协议）。

## 为什么拿满注意力对照局部注意力

| | 进化混合体 | 本基线 |
|--|--|--|
| 注意力 | 滑动窗口 `local_attn`（外加 SSM/GRU/MoE） | 每层满因果注意力 |
| 搜索 | 外环进化 | 固定 GPT 风格栈 |
| 角色 | 研究对象 | 参数预算下的参考线 |

满注意力对「把提示绑到答案」是成熟、强的归纳偏置。这**并不**宣称它在任何嵌入式 / APU 设定下都最优。

## 配置

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
python -m baselines.transformer.train \
  --config baselines/transformer/configs/train_7m_modern.yaml
```

典型规模：`d_model=256`，`n_layers=7`，`n_heads=8` → 约 6.92M 参数（词嵌入共享）。

产物：`runs/train_transformer_baseline_v1/`。
