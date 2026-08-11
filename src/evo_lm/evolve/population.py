"""Population-based evolutionary search over hybrid LM genomes."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional

import torch

from evo_lm.arch.hybrid import (
    Genome,
    build_model,
    crossover,
    enforce_arch_constraints,
    enforce_param_budget,
    estimate_params,
)


FitnessFn = Callable[[Genome], float]


@dataclass
class EvolveConfig:
    population: int = 16
    generations: int = 20
    elite: int = 2
    tournament_k: int = 3
    mutate_rate: float = 0.8
    crossover_rate: float = 0.5
    max_params: int = 500_000_000  # default research scale; configs raise toward <10B
    d_model: int = 256
    vocab_size: int = 8000
    max_seq_len: int = 256
    seed: int = 42
    out_dir: str = "runs/evolve"
    min_local_attn: int = 1
    min_moe: int = 1


def tournament_select(pop: List[Genome], k: int, rng: random.Random) -> Genome:
    contenders = rng.sample(pop, min(k, len(pop)))
    return max(contenders, key=lambda g: g.fitness)


def _genome_key(genome: Genome) -> str:
    payload = genome.to_dict()
    for field in ("generation", "fitness", "n_params"):
        payload.pop(field, None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def evaluate_population(
    pop: List[Genome],
    fitness_fn: FitnessFn,
    cache: dict[str, float] | None = None,
) -> None:
    for g in pop:
        key = _genome_key(g)
        if cache is not None and key in cache:
            g.fitness = cache[key]
        else:
            g.fitness = float(fitness_fn(g))
            if cache is not None:
                cache[key] = g.fitness


def run_evolution(cfg: EvolveConfig, fitness_fn: FitnessFn) -> Genome:
    if cfg.population < 2 or cfg.generations < 1:
        raise ValueError("Evolution requires population >= 2 and generations >= 1")
    if not 1 <= cfg.elite < cfg.population:
        raise ValueError("elite must be in [1, population)")
    if cfg.tournament_k < 1:
        raise ValueError("tournament_k must be positive")

    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pop: List[Genome] = []
    for _ in range(cfg.population):
        g = Genome.random(
            rng,
            d_model=cfg.d_model,
            vocab_size=cfg.vocab_size,
            max_seq_len=cfg.max_seq_len,
        )
        g = enforce_arch_constraints(
            g,
            rng,
            min_local_attn=cfg.min_local_attn,
            min_moe=cfg.min_moe,
        )
        g = enforce_param_budget(
            g,
            cfg.max_params,
            rng,
            min_local_attn=cfg.min_local_attn,
            min_moe=cfg.min_moe,
        )
        pop.append(g)

    history = []
    fitness_cache: dict[str, float] = {}
    best: Optional[Genome] = None
    t0 = time.time()

    for gen in range(cfg.generations):
        evaluate_population(pop, fitness_fn, fitness_cache)
        pop.sort(key=lambda g: g.fitness, reverse=True)
        if best is None or pop[0].fitness > best.fitness:
            best = Genome.from_dict(pop[0].to_dict())
            best.save(str(out / "best_genome.json"))

        avg = sum(g.fitness for g in pop) / len(pop)
        history.append(
            {
                "generation": gen,
                "best_fitness": pop[0].fitness,
                "avg_fitness": avg,
                "best_params": pop[0].n_params,
                "best_blocks": [b.kind for b in pop[0].blocks],
                "population": [g.to_dict() for g in pop],
            }
        )
        pct = 100.0 * (gen + 1) / max(cfg.generations, 1)
        print(
            f"[gen {gen:03d}/{cfg.generations:03d} {pct:5.1f}%] "
            f"best={pop[0].fitness:.4f} avg={avg:.4f} "
            f"params={pop[0].n_params:,} blocks={len(pop[0].blocks)} "
            f"kinds={[b.kind for b in pop[0].blocks]}"
        )

        if gen == cfg.generations - 1:
            break

        # next generation
        next_pop: List[Genome] = [Genome.from_dict(g.to_dict()) for g in pop[: cfg.elite]]
        while len(next_pop) < cfg.population:
            if rng.random() < cfg.crossover_rate:
                p1 = tournament_select(pop, cfg.tournament_k, rng)
                p2 = tournament_select(pop, cfg.tournament_k, rng)
                child = crossover(p1, p2, rng)
            else:
                child = Genome.from_dict(tournament_select(pop, cfg.tournament_k, rng).to_dict())
            if rng.random() < cfg.mutate_rate:
                child = child.mutate(rng)
            child = enforce_arch_constraints(
                child,
                rng,
                min_local_attn=cfg.min_local_attn,
                min_moe=cfg.min_moe,
            )
            child = enforce_param_budget(
                child,
                cfg.max_params,
                rng,
                min_local_attn=cfg.min_local_attn,
                min_moe=cfg.min_moe,
            )
            next_pop.append(child)
        pop = next_pop

    with open(out / "history.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "history": history,
                "elapsed_sec": time.time() - t0,
                "unique_evaluations": len(fitness_cache),
            },
            f,
            indent=2,
        )
    assert best is not None
    return best


def quick_fitness_proxy(
    genome: Genome,
    *,
    device: torch.device,
    batches: int = 3,
    seq_len: int = 64,
    steps: int = 5,
    lr: float = 3e-3,
    use_vulkan: bool | None = None,
    seed: int = 42,
    moe_aux_coef: float = 0.01,
) -> float:
    """Cheap smoke proxy on deterministic, learnable token transitions.

    This only checks trainability. Real searches should use corpus-backed fitness.
    Higher is better: negative loss.
    """
    import os

    if estimate_params(genome) <= 0:
        return -1e9
    if use_vulkan is None:
        use_vulkan = os.environ.get("EVO_DEVICE", "vulkan").lower() not in {"cpu"}
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
        total = 0.0
        usable_vocab = max(1, genome.vocab_size - 4)
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        for step in range(steps):
            starts = torch.arange(batches, device=device).unsqueeze(1) + step * batches
            idx = 4 + (starts + positions) % usable_vocab
            _, loss = model(idx, idx)
            if loss is None:
                return -1e9
            objective = loss + moe_aux_coef * model.moe_aux_loss
            if not torch.isfinite(objective):
                return -1e9
            opt.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
        return -total / steps
