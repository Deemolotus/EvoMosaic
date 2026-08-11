import json
from pathlib import Path

import torch
from torch import nn

from evo_lm.arch.hybrid import HybridLM
from evo_lm.data.modern import (
    SourceGroup,
    add_compositional_variants,
    build_modern_mix,
)


def _hsk_entry(word: str, pinyin: str, pos: str, frequency: int) -> dict:
    return {
        "simplified": word,
        "frequency": frequency,
        "pos": [pos],
        "forms": [{"transcriptions": {"pinyin": pinyin}}],
    }


def test_compositional_tasks_keep_natural_sentence_as_target():
    sentence = "我们今天在安静的教室里认真学习汉语知识。"
    groups = add_compositional_variants(
        [SourceGroup("sample", "fixture", [f"用户: 怎么学习？\n助手: {sentence}"])],
        [
            {"word": "学习"},
            {"word": "汉语"},
            {"word": "知识"},
        ],
        seed=7,
        combination_rate=1.0,
        reorder_rate=1.0,
    )
    blocks = groups[0].blocks
    assert any("组成一句自然的话" in block for block in blocks)
    assert any("排列成自然句子" in block for block in blocks)
    assert all(block.endswith(sentence) for block in blocks)


def test_modern_builder_writes_source_aware_splits(tmp_path: Path):
    legacy = tmp_path / "legacy.txt"
    legacy.write_text(
        "用户: 你好\n助手: 你好，有什么可以帮你？\n\n"
        "春天来了，树木发出新芽。\n",
        encoding="utf-8",
    )
    hc3 = tmp_path / "hc3.jsonl"
    hc3.write_text(
        json.dumps(
            {
                "question": "怎样保持每天阅读的习惯？",
                "human_answers": ["每天固定读十分钟，从容易读完的小文章开始，慢慢增加时间。"],
                "chatgpt_answers": ["不应被采用。"],
                "source": "baike",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    coig = tmp_path / "coig.jsonl"
    coig.write_text(
        "\n".join(
            json.dumps(
                {
                    "trans_instruction": f"请简要说明第{i}个学习方法。",
                    "trans_input": "制定清晰的小目标。",
                    "trans_output": "先完成一个具体的小目标，再根据结果调整下一步计划。",
                },
                ensure_ascii=False,
            )
            for i in range(8)
        )
        + "\n",
        encoding="utf-8",
    )
    hsk = tmp_path / "hsk.json"
    hsk.write_text(
        json.dumps(
            [
                _hsk_entry("学习", "xué xí", "v", 1),
                _hsk_entry("目标", "mù biāo", "n", 2),
                _hsk_entry("计划", "jì huà", "n", 3),
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    coig_value = tmp_path / "coig_value.json"
    coig_value.write_text(
        json.dumps(
            [
                {
                    "instruction": "朋友情绪不好时应该怎么办？",
                    "input": "",
                    "output": "先耐心听对方说完，再询问对方现在需要陪伴还是具体建议。",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    manifest_path = tmp_path / "manifest.json"
    manifest = build_modern_mix(
        legacy_path=legacy,
        hc3_path=hc3,
        coig_path=coig,
        coig_value_path=coig_value,
        hsk_path=hsk,
        train_out=train,
        val_out=val,
        manifest_out=manifest_path,
        seed=11,
        val_ratio=0.3,
        coig_limit=8,
        hsk_limit=3,
    )
    train_blocks = set(train.read_text(encoding="utf-8").strip().split("\n\n"))
    val_blocks = set(val.read_text(encoding="utf-8").strip().split("\n\n"))
    assert train_blocks
    assert val_blocks
    assert train_blocks.isdisjoint(val_blocks)
    assert manifest["train_blocks"] == len(train_blocks)
    assert manifest["val_blocks"] == len(val_blocks)
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["seed"] == 11


class _EosDummy(nn.Module):
    generate = HybridLM.generate

    def __init__(self) -> None:
        super().__init__()
        self.genome = type("GenomeStub", (), {"max_seq_len": 8})()

    def forward(self, idx, targets=None):
        logits = torch.full((*idx.shape, 6), -100.0, device=idx.device)
        logits[..., 3] = 100.0
        return logits, None


def test_generation_stops_at_eos():
    torch.manual_seed(1)
    model = _EosDummy()
    prompt = torch.tensor([[2, 4]])
    output = model.generate(prompt, max_new_tokens=20, top_k=1, eos_token_id=3)
    assert output.tolist() == [[2, 4, 3]]
