"""CLI entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml


def prepare() -> None:
    parser = argparse.ArgumentParser(description="Prepare classical corpus + LaTeX + dialogue")
    parser.add_argument("--raw", default="data/raw/poetry")
    parser.add_argument("--out-text", default="data/processed/corpus.txt")
    parser.add_argument("--latex-dir", default="data/latex")
    parser.add_argument("--dialogue", default="data/processed/dialogue.txt")
    parser.add_argument("--max-vocab", type=int, default=12000)
    args = parser.parse_args()

    from evo_lm.data.prepare import export_all_latex, json_corpus_to_text, write_dialogue_corpus
    from evo_lm.data.tokenizer import CharTokenizer

    raw = Path(args.raw)
    n = json_corpus_to_text(raw, Path(args.out_text))
    write_dialogue_corpus(Path(args.dialogue))
    tex_files = export_all_latex(raw, Path(args.latex_dir))
    text = Path(args.out_text).read_text(encoding="utf-8") + "\n" + Path(args.dialogue).read_text(encoding="utf-8")
    tok = CharTokenizer.train_from_texts([text], max_vocab=args.max_vocab)
    tok_path = Path("data/processed/tokenizer.json")
    tok.save(tok_path)
    print(f"records≈{n}, latex_files={len(tex_files)}, vocab={tok.vocab_size}")
    print(f"text -> {args.out_text}")
    print(f"dialogue -> {args.dialogue}")
    print(f"latex -> {args.latex_dir}")
    print(f"tokenizer -> {tok_path}")


def evolve() -> None:
    parser = argparse.ArgumentParser(description="Evolve hybrid architectures")
    parser.add_argument("--config", default="configs/evolve_nano.yaml")
    args = parser.parse_args()

    import os
    import statistics
    from evo_lm.data.dataset import build_dataloaders, load_text_files, text_blocks
    from evo_lm.data.tokenizer import CharTokenizer
    from evo_lm.eval.fitness import corpus_fitness, parse_compositional_tasks
    from evo_lm.evolve.population import EvolveConfig, quick_fitness_proxy, run_evolution
    from evo_lm.runtime.device import detect_device

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg.get("allow_cpu") or cfg.get("evolve_on_host"):
        os.environ["EVO_ALLOW_CPU"] = "1"
    # Fitness loops are many; host BLAS is much faster than sync Vulkan GEMM.
    if cfg.get("evolve_on_host"):
        os.environ["EVO_DEVICE"] = "cpu"
        prefer = "cpu"
    else:
        prefer = cfg.get("device", "vulkan")

    device_info = detect_device(prefer)
    device = device_info.torch_device
    print(device_info)
    evo_cfg = EvolveConfig(**cfg["evolve"])
    use_vk = device_info.backend == "vulkan" and not cfg.get("evolve_on_host")

    mode = cfg.get("fitness", "proxy")
    fitness_seed = int(cfg.get("fitness_seed", evo_cfg.seed))
    fitness_repeats = max(1, int(cfg.get("fitness_repeats", 1)))
    moe_aux_coef = float(cfg.get("moe_aux_coef", 0.01))
    if mode == "corpus":
        text = load_text_files([Path(p) for p in cfg["data"]["texts"]])
        tok_path = Path(cfg["data"]["tokenizer"])
        if cfg["data"].get("rebuild_tokenizer") or not tok_path.exists():
            tok = CharTokenizer.train_from_texts(
                [text],
                max_vocab=int(cfg["data"].get("max_vocab", 12000)),
            )
            tok_path.parent.mkdir(parents=True, exist_ok=True)
            tok.save(tok_path)
            print(f"[data] rebuilt tokenizer: {tok_path} vocab={tok.vocab_size}")
        else:
            tok = CharTokenizer.load(tok_path)
        val_paths = [Path(p) for p in cfg["data"].get("val_texts", [])]
        val_text = load_text_files(val_paths) if val_paths else None
        pack_mode = str(cfg["data"].get("pack_mode", "dialogue"))
        assistant_loss_only = bool(cfg["data"].get("assistant_loss_only", True))
        train_loader, val_loader = build_dataloaders(
            text,
            tok,
            seq_len=cfg.get("seq_len", 128),
            batch_size=cfg.get("batch_size", 4),
            val_ratio=float(cfg["data"].get("val_ratio", 0.05)),
            val_text=val_text,
            split_seed=int(cfg["data"].get("split_seed", evo_cfg.seed)),
            shuffle_train=False,
            fixed_train_order=True,
            pack_mode=pack_mode,
            assistant_loss_only=assistant_loss_only,
        )
        evo_cfg.vocab_size = tok.vocab_size

        composition_items: list[dict] = []
        scramble_items: list[dict] = []
        task_source = val_text if val_text else text
        composition_items, scramble_items = parse_compositional_tasks(text_blocks(task_source))
        kw_weight = float(cfg.get("kw_weight", 0.35))
        scramble_weight = float(cfg.get("scramble_weight", 0.45))
        max_composition = int(cfg.get("max_composition_eval", 12))
        max_scramble = int(cfg.get("max_scramble_eval", 12))
        print(
            f"[fitness] pack_mode={pack_mode} assistant_loss_only={assistant_loss_only} "
            f"composition={len(composition_items)} scramble={len(scramble_items)} "
            f"kw_w={kw_weight} scramble_w={scramble_weight}"
        )

        def fitness_fn(g):
            g.vocab_size = tok.vocab_size
            # force host linear inside fitness via env for speed
            old = os.environ.get("EVO_DEVICE")
            if cfg.get("evolve_on_host"):
                os.environ["EVO_DEVICE"] = "cpu"
            try:
                scores = [
                    corpus_fitness(
                        g,
                        train_loader,
                        val_loader,
                        device=device,
                        train_steps=cfg.get("train_steps", 30),
                        lr=float(cfg.get("lr", 1e-3)),
                        max_params=evo_cfg.max_params,
                        use_vulkan=use_vk,
                        seed=fitness_seed + repeat,
                        moe_aux_coef=moe_aux_coef,
                        tokenizer=tok,
                        composition_items=composition_items,
                        scramble_items=scramble_items,
                        kw_weight=kw_weight,
                        scramble_weight=scramble_weight,
                        max_composition=max_composition,
                        max_scramble=max_scramble,
                    )
                    for repeat in range(fitness_repeats)
                ]
                return statistics.median(scores)
            finally:
                if old is not None:
                    os.environ["EVO_DEVICE"] = old

    else:

        def fitness_fn(g):
            scores = [
                quick_fitness_proxy(
                    g,
                    device=device,
                    use_vulkan=use_vk,
                    seed=fitness_seed + repeat,
                    moe_aux_coef=moe_aux_coef,
                )
                for repeat in range(fitness_repeats)
            ]
            return statistics.median(scores)

    best = run_evolution(evo_cfg, fitness_fn)
    print("best genome saved; fitness=", best.fitness, "params=", best.n_params)
    print("blocks=", [b.kind for b in best.blocks])
    assert any(b.kind == "local_attn" for b in best.blocks), "best genome missing local_attn"
    assert any(b.kind == "moe" for b in best.blocks), "best genome missing moe"


def train() -> None:
    parser = argparse.ArgumentParser(description="Train evolved / seeded hybrid LM")
    parser.add_argument("--config", default="configs/train_nano.yaml")
    args = parser.parse_args()
    from evo_lm.train import train_from_config

    ckpt = train_from_config(args.config)
    print("checkpoint:", ckpt)


def chat() -> None:
    parser = argparse.ArgumentParser(description="Generate / chat with a checkpoint")
    parser.add_argument("--ckpt", default="runs/train_speak/best.pt")
    parser.add_argument("--tokenizer", default="data/processed/tokenizer_speak.json")
    parser.add_argument("--prompt", default="用户: 你好\n助手:")
    parser.add_argument("--tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    from evo_lm.arch.hybrid import Genome, build_model
    from evo_lm.data.tokenizer import CharTokenizer
    from evo_lm.runtime.device import detect_device
    import os

    os.environ.setdefault("EVO_ALLOW_CPU", "1")
    device_info = detect_device(os.environ.get("EVO_DEVICE", "cpu"))
    device = device_info.torch_device
    tok = CharTokenizer.load(args.tokenizer)
    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    genome = Genome.from_dict(blob["genome"])
    genome.vocab_size = tok.vocab_size
    model = build_model(genome, device=device, use_vulkan=False)
    model.load_state_dict(blob["model"], strict=False)
    ids = torch.tensor([tok.encode(args.prompt, add_bos=True)], device=device)
    out = model.generate(ids, max_new_tokens=args.tokens, temperature=args.temperature, top_k=args.top_k)
    text = tok.decode(out[0].tolist())
    # Prefer printing from the dialogue markers when present.
    if "助手:" in text:
        print(text[text.find("用户:"):] if "用户:" in text else text)
    else:
        print(text)


def export_latex() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/poetry")
    parser.add_argument("--out", default="data/latex")
    args = parser.parse_args()
    from evo_lm.data.prepare import export_all_latex

    files = export_all_latex(Path(args.raw), Path(args.out))
    print(f"wrote {len(files)} latex files to {args.out}")


if __name__ == "__main__":
    prepare()
