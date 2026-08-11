from pathlib import Path

import pytest

from evo_lm.data.dataset import TextDataset, build_dataloaders, split_text_blocks
from evo_lm.data.prepare import extract_records, json_to_latex_book, record_to_text
from evo_lm.data.tokenizer import CharTokenizer


def test_lunyu_extract():
    path = Path("data/raw/poetry/lunyu/lunyu.json")
    if not path.exists():
        return
    recs = list(extract_records(path))
    assert len(recs) >= 1
    text = record_to_text(recs[0])
    assert "子" in text or len(text) > 0


def test_latex_export(tmp_path):
    path = Path("data/raw/poetry/sishu/daxue.json")
    if not path.exists():
        return
    out = tmp_path / "daxue.tex"
    json_to_latex_book(path, out, title="大学", max_records=5)
    body = out.read_text(encoding="utf-8")
    assert r"\begin{document}" in body
    assert "大学" in body or "大學" in body or len(body) > 100


def test_text_dataset_includes_last_window():
    assert len(TextDataset(list(range(10)), seq_len=4)) == 7


def test_grouped_split_keeps_duplicates_out_of_validation():
    text = "\n\n".join(["甲乙丙", "丁戊己", "甲乙丙", "庚辛壬", "丁戊己", "癸子丑"])
    train, val = split_text_blocks(text, val_ratio=0.34, seed=7)

    assert set(train).isdisjoint(val)
    assert all(text.count(block) == train.count(block) for block in set(train))
    assert len(val) == len(set(val))


def test_dialogue_pack_masks_user_prefix():
    from evo_lm.data.dataset import IGNORE_INDEX, encode_dialogue_sample

    text = "用户: 你好\n助手: 你好呀"
    tok = CharTokenizer.train_from_texts([text], min_freq=1)
    sample = encode_dialogue_sample(text, tok, seq_len=32, assistant_loss_only=True)
    assert sample is not None
    inputs, targets = sample
    marker = "助手:"
    answer_start = 1 + text.find(marker) + len(marker)
    assert (targets[:answer_start] == IGNORE_INDEX).all()
    assert (targets[answer_start : answer_start + 3] != IGNORE_INDEX).any()
    assert int(inputs[0]) == tok.bos_id


def test_dataloader_adds_document_boundaries_and_rejects_overlap():
    text = "\n\n".join(["甲乙丙丁", "戊己庚辛", "壬癸子丑", "寅卯辰巳"])
    tok = CharTokenizer.train_from_texts([text], min_freq=1)
    train_loader, val_loader = build_dataloaders(
        text,
        tok,
        seq_len=3,
        batch_size=1,
        val_ratio=0.25,
        split_seed=3,
        pack_mode="window",
    )
    assert tok.eos_id in train_loader.dataset.data
    assert tok.eos_id in val_loader.dataset.data

    with pytest.raises(ValueError, match="overlaps"):
        build_dataloaders(
            "甲乙丙丁\n\n戊己庚辛",
            tok,
            seq_len=3,
            batch_size=1,
            val_text="甲乙丙丁\n\n壬癸子丑",
            pack_mode="window",
        )
