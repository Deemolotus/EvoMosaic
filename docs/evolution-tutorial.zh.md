# 进化架构搜索如何工作

[English](evolution-tutorial.md)

本项目用的是 **evolutionary NAS**：外环搜索*结构*（堆哪些块），内环用梯度下降训练*权重*。

## 基因组 → 表型

候选个体是一个 `Genome`（`src/evo_lm/arch/hybrid.py`）：

```text
Genome
├── d_model, vocab_size, max_seq_len
└── blocks[]  (BlockGene 序列)
      kind ∈ {ssm, gru, local_attn, conv, moe, mlp}
      expand, d_state, n_heads, window, n_experts, …
```

由基因组构建 `HybridLM` 就是表型。软先验更倾向 SSM/GRU/卷积；**硬约束**仍要求：

- 至少一个 `local_attn`
- 至少一个 `moe`

## 外环循环

```text
初始化种群  →  评估适应度  →  保留精英
      ↑                │
      │                ↓
      └── 交叉 / 变异（锦标赛选择父母）
```

实现：`src/evo_lm/evolve/population.py` → `run_evolution`。

每次适应度评估（`corpus` 模式，`src/evo_lm/eval/fitness.py`）：

1. 在设备上建模型；
2. 在现代对话数据上短训（只对「助手答案」算主损失）；
3. 在验证答案 token 上量 NLL；
4. 对少量组句 / 乱序提示生成，算关键词覆盖与精确匹配；
5. 大致返回：

```text
fitness = -val_NLL
        + λ_kw * keyword_coverage
        + λ_sc * scramble_exact_match
        - size_penalty
```

越大越好。超出参数预算的基因组直接丢弃。

## 适应度例题

假设短训后：

- 验证 NLL = 6.4  
- 关键词覆盖 = 0.25  
- 乱序精确率 = 0.0  
- 参数量 ≈ 7e6，尺寸惩罚 ≈ 0.00035  

取 `λ_kw=0.35`，`λ_sc=0.45`：

```text
fitness ≈ -6.4 + 0.35*0.25 + 0.45*0.0 - 0.00035 ≈ -6.31
```

降低 NLL、提高组合任务分数都会推动种群前进。

## 命令

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0   # AMD Radeon 780M / gfx1103
python -m evo_lm evolve --config configs/evolve_modern_moe_attn_v1.yaml
```

优胜者：`runs/train_modern_moe_attn_v1/best_genome.json`（与训练权重放在同一目录）。

```bash
python -m evo_lm train --config configs/train_modern_moe_attn_v1.yaml
```

## 进化做不到的事

它**不会**自动学会你从未写进适应度的能力。本轮实验中乱序精确率一直是 0——搜索奖励的是更低的 LM 损失，以及更常命中指定词，而不是完美的排列复原器。
