"""Simple character-level tokenizer for classical Chinese + dialogue.

SentencePiece is optional for larger runs; char-level is transparent for learning.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence


SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>"]


class CharTokenizer:
    def __init__(self, stoi: dict[str, int]) -> None:
        self.stoi = stoi
        self.itos = {i: s for s, i in stoi.items()}
        self.pad_id = stoi["<pad>"]
        self.unk_id = stoi["<unk>"]
        self.bos_id = stoi["<bos>"]
        self.eos_id = stoi["<eos>"]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for ch in text:
            ids.append(self.stoi.get(ch, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        out = []
        for i in ids:
            ch = self.itos.get(int(i), "")
            if ch in SPECIAL:
                continue
            out.append(ch)
        return "".join(out)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.stoi, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "CharTokenizer":
        stoi = json.loads(Path(path).read_text(encoding="utf-8"))
        return CharTokenizer(stoi)

    @staticmethod
    def train_from_texts(texts: Iterable[str], max_vocab: int = 12000, min_freq: int = 2) -> "CharTokenizer":
        ctr: Counter[str] = Counter()
        for t in texts:
            ctr.update(t)
        chars = [c for c, n in ctr.most_common() if n >= min_freq]
        chars = chars[: max(0, max_vocab - len(SPECIAL))]
        stoi = {s: i for i, s in enumerate(SPECIAL + chars)}
        return CharTokenizer(stoi)
