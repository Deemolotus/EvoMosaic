"""Torch dataset utilities."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from evo_lm.data.tokenizer import CharTokenizer

IGNORE_INDEX = -100
# Chinese dialogue protocol marker; loss starts after this span.
ASSISTANT_MARKER = "助手:"


class TextDataset(Dataset):
    def __init__(self, token_ids: List[int], seq_len: int) -> None:
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, self.data.numel() - self.seq_len + 1)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx : idx + self.seq_len]


class DialogueLMDataset(Dataset):
    """One dialogue/document block per sample with optional assistant-answer loss mask."""

    def __init__(
        self,
        blocks: list[str],
        tokenizer: CharTokenizer,
        seq_len: int,
        *,
        assistant_loss_only: bool = True,
    ) -> None:
        self.seq_len = seq_len
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for block in blocks:
            encoded = encode_dialogue_sample(
                block,
                tokenizer,
                seq_len,
                assistant_loss_only=assistant_loss_only,
            )
            if encoded is not None:
                self.samples.append(encoded)
        if not self.samples:
            raise ValueError("DialogueLMDataset produced zero usable samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


class FixedRandomSampler(Sampler[int]):
    """Return the same seeded random permutation on every iteration."""

    def __init__(self, data_source: Dataset, seed: int) -> None:
        self.data_source = data_source
        self.seed = seed

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        return iter(torch.randperm(len(self.data_source), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.data_source)


def load_text_files(paths: List[Path]) -> str:
    chunks = []
    for p in paths:
        if p.exists():
            chunks.append(p.read_text(encoding="utf-8"))
    return "\n\n".join(chunks)


def text_blocks(text: str) -> list[str]:
    """Return document/dialogue blocks separated by one or more blank lines."""
    return [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]


# Back-compat alias for internal callers.
_text_blocks = text_blocks


def split_text_blocks(
    text: str,
    *,
    val_ratio: float,
    seed: int = 42,
    min_val_chars: int = 0,
) -> tuple[list[str], list[str]]:
    """Split by exact-content groups so duplicates never cross train/val.

    Training retains repeated blocks as intentional upsampling. Validation gets
    one copy of each selected unique block, which prevents a repeated corpus
    suffix from masquerading as held-out data.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    blocks = _text_blocks(text)
    unique = list(dict.fromkeys(blocks))
    if len(unique) < 2:
        raise ValueError("Need at least two distinct text blocks for a leakage-free split")

    rng = random.Random(seed)
    rng.shuffle(unique)
    target_chars = max(min_val_chars, int(sum(map(len, unique)) * val_ratio))
    val_blocks: list[str] = []
    val_chars = 0
    for block in unique[:-1]:  # always leave at least one unique training block
        val_blocks.append(block)
        val_chars += len(block)
        if val_chars >= target_chars:
            break

    val_keys = set(val_blocks)
    train_blocks = [block for block in blocks if block not in val_keys]
    if not train_blocks or not val_blocks:
        raise ValueError("Could not produce non-empty train and validation splits")
    return train_blocks, val_blocks


def _encode_blocks(blocks: list[str], tokenizer: CharTokenizer) -> list[int]:
    ids: list[int] = []
    for block in blocks:
        ids.extend(tokenizer.encode(block, add_bos=True, add_eos=True))
    return ids


def encode_dialogue_sample(
    text: str,
    tokenizer: CharTokenizer,
    seq_len: int,
    *,
    assistant_loss_only: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Encode one block; optionally mask CE loss through the assistant marker."""
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    if len(ids) < 2:
        return None

    answer_start = 0
    if assistant_loss_only:
        marker_pos = text.find(ASSISTANT_MARKER)
        if marker_pos >= 0:
            # BOS + chars up to and including the marker; loss starts on the first answer char.
            answer_start = 1 + marker_pos + len(ASSISTANT_MARKER)
            answer_start = min(answer_start, len(ids) - 1)

    if len(ids) > seq_len:
        # Prefer keeping the answer tail when truncating long dialogues.
        overflow = len(ids) - seq_len
        ids = ids[overflow:]
        answer_start = max(0, answer_start - overflow)

    n = len(ids)
    if n < seq_len:
        ids = ids + [tokenizer.pad_id] * (seq_len - n)

    input_ids = torch.tensor(ids, dtype=torch.long)
    targets = input_ids.clone()
    if assistant_loss_only and answer_start > 0:
        targets[:answer_start] = IGNORE_INDEX
    targets[input_ids == tokenizer.pad_id] = IGNORE_INDEX
    # CE uses targets[:, 1:]; need at least one supervised answer token.
    if int((targets[1:] != IGNORE_INDEX).sum()) == 0:
        return None
    return input_ids, targets


def unpack_lm_batch(batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Support both window tensors and (input, target) dialogue batches."""
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        return batch[0], batch[1]
    return batch, batch


def build_dataloaders(
    text: str,
    tokenizer: CharTokenizer,
    *,
    seq_len: int = 256,
    batch_size: int = 8,
    val_ratio: float = 0.05,
    num_workers: int = 0,
    pin_memory: bool = False,
    val_text: str | None = None,
    split_seed: int = 42,
    shuffle_train: bool = True,
    fixed_train_order: bool = False,
    pack_mode: str = "window",
    assistant_loss_only: bool = True,
) -> tuple[DataLoader, DataLoader]:
    if val_text is None:
        train_blocks, val_blocks = split_text_blocks(
            text,
            val_ratio=val_ratio,
            seed=split_seed,
            min_val_chars=seq_len + 1,
        )
    else:
        train_blocks = _text_blocks(text)
        val_blocks = list(dict.fromkeys(_text_blocks(val_text)))
        overlap = set(train_blocks) & set(val_blocks)
        if overlap:
            raise ValueError(f"Explicit validation data overlaps training data in {len(overlap)} blocks")

    if pack_mode not in {"window", "dialogue"}:
        raise ValueError(f"Unknown pack_mode={pack_mode!r}; expected 'window' or 'dialogue'")

    if pack_mode == "dialogue":
        train_ds: Dataset = DialogueLMDataset(
            train_blocks,
            tokenizer,
            seq_len,
            assistant_loss_only=assistant_loss_only,
        )
        val_ds: Dataset = DialogueLMDataset(
            val_blocks,
            tokenizer,
            seq_len,
            assistant_loss_only=assistant_loss_only,
        )
    else:
        train_ids = _encode_blocks(train_blocks, tokenizer)
        val_ids = _encode_blocks(val_blocks, tokenizer)
        train_ds = TextDataset(train_ids, seq_len)
        val_ds = TextDataset(val_ids, seq_len)
        if len(train_ds) == 0 or len(val_ds) == 0:
            raise ValueError(
                f"Not enough tokens for seq_len={seq_len}: "
                f"train={len(train_ids)}, val={len(val_ids)}"
            )

    # CPU workers prefetch batches while the GPU trains (useful CPU+GPU overlap).
    common = dict(
        batch_size=batch_size,
        num_workers=max(0, int(num_workers)),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(num_workers and num_workers > 0),
    )
    if common["num_workers"] > 0:
        common["prefetch_factor"] = 2
    sampler = FixedRandomSampler(train_ds, split_seed) if fixed_train_order else None
    generator = torch.Generator().manual_seed(split_seed) if shuffle_train and sampler is None else None
    train_loader = DataLoader(
        train_ds,
        shuffle=shuffle_train and sampler is None,
        sampler=sampler,
        drop_last=True,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader
