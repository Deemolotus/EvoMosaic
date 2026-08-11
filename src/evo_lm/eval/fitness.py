"""Corpus-backed fitness: short fine-tune then measure validation NLL + task metrics."""

from __future__ import annotations

import re
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader

from evo_lm.arch.hybrid import Genome, build_model, estimate_params
from evo_lm.data.dataset import IGNORE_INDEX, unpack_lm_batch
from evo_lm.data.tokenizer import CharTokenizer

COMPOSITION_RE = re.compile(r'请用[“"](.+?)[”"]和[“"](.+?)[”"]组成一句')
SCRAMBLE_RE = re.compile(r"请把这些片段排列成自然句子：")
# Above patterns match the Chinese task prompts written by modern.py.


def unpack_batch(batch) -> tuple[torch.Tensor, torch.Tensor]:
    return unpack_lm_batch(batch)


@torch.no_grad()
def eval_nll(model, loader: DataLoader, device: torch.device, max_batches: int = 20) -> float:
    model.eval()
    total_loss = 0.0
    n_tokens = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        inputs, targets = unpack_batch(batch)
        inputs = inputs.to(device)
        targets = targets.to(device)
        _, loss = model(inputs, targets)
        if loss is None or not torch.isfinite(loss):
            continue
        tokens = int((targets[:, 1:] != IGNORE_INDEX).sum().item())
        if tokens <= 0:
            continue
        total_loss += float(loss) * tokens
        n_tokens += tokens
    if n_tokens == 0:
        return 1e9
    return total_loss / n_tokens


def parse_compositional_tasks(blocks: Sequence[str]) -> tuple[list[dict], list[dict]]:
    """Extract keyword-composition and scramble eval items from dialogue blocks."""
    composition: list[dict] = []
    scramble: list[dict] = []
    for block in blocks:
        if "\n助手:" not in block:
            continue
        user, answer = block.split("\n助手:", 1)
        answer = answer.lstrip()
        question = user
        if question.startswith("用户:"):
            question = question[len("用户:") :].lstrip()
        prompt = f"{user}\n助手:"
        m = COMPOSITION_RE.search(question)
        if m:
            composition.append(
                {
                    "prompt": prompt,
                    "keywords": [m.group(1), m.group(2)],
                    "gold": answer,
                }
            )
            continue
        if SCRAMBLE_RE.search(question):
            scramble.append({"prompt": prompt, "gold": answer})
    return composition, scramble


def _assistant_completion(text: str) -> str:
    if "助手:" in text:
        return text.split("助手:", 1)[1].strip()
    return text.strip()


def keyword_coverage(completion: str, keywords: Sequence[str]) -> float:
    if not keywords:
        return 0.0
    hits = sum(1 for kw in keywords if kw and kw in completion)
    return hits / len(keywords)


def scramble_exact_match(completion: str, gold: str) -> float:
    pred = completion.strip().rstrip("。！？!?；;")
    target = gold.strip().rstrip("。！？!?；;")
    if not target:
        return 0.0
    if pred == target or pred.startswith(target) or target in pred:
        return 1.0
    return 0.0


@torch.no_grad()
def eval_task_metrics(
    model,
    tokenizer: CharTokenizer,
    composition_items: Sequence[dict],
    scramble_items: Sequence[dict],
    *,
    device: torch.device,
    max_composition: int = 24,
    max_scramble: int = 24,
    max_new_tokens: int = 48,
) -> tuple[float, float]:
    """Return (keyword_coverage, scramble_exact_match) in [0, 1]."""
    model.eval()
    kw_scores: list[float] = []
    for item in list(composition_items)[:max_composition]:
        ids = torch.tensor([tokenizer.encode(item["prompt"], add_bos=True)], device=device)
        out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.2, top_k=20)
        text = tokenizer.decode(out[0].tolist())
        completion = _assistant_completion(text)
        kw_scores.append(keyword_coverage(completion, item["keywords"]))

    sc_scores: list[float] = []
    for item in list(scramble_items)[:max_scramble]:
        ids = torch.tensor([tokenizer.encode(item["prompt"], add_bos=True)], device=device)
        out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.1, top_k=10)
        text = tokenizer.decode(out[0].tolist())
        completion = _assistant_completion(text)
        sc_scores.append(scramble_exact_match(completion, item["gold"]))

    kw = sum(kw_scores) / len(kw_scores) if kw_scores else 0.0
    sc = sum(sc_scores) / len(sc_scores) if sc_scores else 0.0
    return kw, sc


def corpus_fitness(
    genome: Genome,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device,
    train_steps: int = 50,
    lr: float = 1e-3,
    max_params: Optional[int] = None,
    use_vulkan: bool = False,
    seed: int = 42,
    moe_aux_coef: float = 0.01,
    tokenizer: CharTokenizer | None = None,
    composition_items: Sequence[dict] | None = None,
    scramble_items: Sequence[dict] | None = None,
    kw_weight: float = 0.35,
    scramble_weight: float = 0.45,
    max_composition: int = 16,
    max_scramble: int = 16,
) -> float:
    """Higher is better: -val NLL + compositional task bonuses - size penalty."""
    if max_params is not None and estimate_params(genome) > max_params:
        return -1e9
    # Soft rejection if evolution somehow lost required blocks.
    kinds = {b.kind for b in genome.blocks}
    if "local_attn" not in kinds or "moe" not in kinds:
        return -1e9

    fork_devices = [device.index or torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        try:
            model = build_model(genome, device=device, use_vulkan=use_vulkan)
        except Exception:
            return -1e9

        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        model.train()
        it = iter(train_loader)
        for _ in range(train_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)
            inputs, targets = unpack_batch(batch)
            inputs = inputs.to(device)
            targets = targets.to(device)
            _, loss = model(inputs, targets)
            if loss is None:
                return -1e9
            objective = loss + moe_aux_coef * model.moe_aux_loss
            if not torch.isfinite(objective):
                return -1e9
            opt.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        nll = eval_nll(model, val_loader, device)
        kw = 0.0
        scramble = 0.0
        if tokenizer is not None and (composition_items or scramble_items):
            kw, scramble = eval_task_metrics(
                model,
                tokenizer,
                composition_items or [],
                scramble_items or [],
                device=device,
                max_composition=max_composition,
                max_scramble=max_scramble,
            )

    size_penalty = 0.05 * (genome.n_params / 1e9)
    return -nll + kw_weight * kw + scramble_weight * scramble - size_penalty
