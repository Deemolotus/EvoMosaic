"""Train a Transformer LM baseline under the same data/loss protocol as the hybrid run."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

# Ensure repo root + src are importable when launched as a module.
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from evo_lm.data.dataset import build_dataloaders, load_text_files, text_blocks, unpack_lm_batch  # noqa: E402
from evo_lm.data.tokenizer import CharTokenizer  # noqa: E402
from evo_lm.eval.fitness import eval_nll, eval_task_metrics, parse_compositional_tasks  # noqa: E402
from evo_lm.runtime.device import detect_device  # noqa: E402

from baselines.transformer.model import build_transformer_lm  # noqa: E402


def _cosine_lr(step: int, total: int, base: float, min_lr: float) -> float:
    if total <= 1:
        return base
    progress = step / max(1, total - 1)
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def _sample_chat(model, tokenizer, device, prompt: str = "用户: 你好\n助手:", tokens: int = 60) -> str:
    model.eval()
    ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], device=device)
    out = model.generate(ids, max_new_tokens=tokens, temperature=0.8, top_k=30)
    text = tokenizer.decode(out[0].tolist())
    model.train()
    return text


def train_from_config(config_path: str) -> Path:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    threads = int(cfg.get("num_threads") or os.environ.get("EVO_THREADS", "16"))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(max(1, min(4, threads // 4)))

    prefer = cfg.get("device", os.environ.get("EVO_DEVICE", "rocm"))
    if cfg.get("allow_cpu"):
        os.environ["EVO_ALLOW_CPU"] = "1"
    device_info = detect_device(prefer)
    device = device_info.torch_device
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    print(device_info)
    if device_info.backend == "cpu" and not cfg.get("allow_cpu"):
        raise RuntimeError("Refusing CPU-only train; set device: rocm|vulkan or allow_cpu: true.")
    if device_info.backend == "rocm":
        print("[train] ROCm/HIP active → Transformer baseline on GPU.")

    data_cfg = cfg["data"]
    paths = [Path(p) for p in data_cfg["texts"]]
    # Resolve relative to repo root when launched from elsewhere.
    paths = [p if p.is_absolute() else _REPO_ROOT / p for p in paths]
    text = load_text_files(paths)
    if not text.strip():
        raise FileNotFoundError(f"No text loaded from {paths}")

    tok_path = Path(data_cfg.get("tokenizer", "data/processed/tokenizer_modern_v3.json"))
    if not tok_path.is_absolute():
        tok_path = _REPO_ROOT / tok_path
    rebuild_tok = bool(data_cfg.get("rebuild_tokenizer", False))
    if tok_path.exists() and not rebuild_tok:
        tokenizer = CharTokenizer.load(tok_path)
    else:
        tokenizer = CharTokenizer.train_from_texts([text], max_vocab=data_cfg.get("max_vocab", 8000))
        tok_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(tok_path)

    nw = int(cfg["train"].get("num_workers", 0))
    pin = bool(cfg["train"].get("pin_memory", device_info.backend == "rocm"))
    val_paths = [Path(p) for p in data_cfg.get("val_texts", [])]
    val_paths = [p if p.is_absolute() else _REPO_ROOT / p for p in val_paths]
    val_text = load_text_files(val_paths) if val_paths else None
    pack_mode = str(data_cfg.get("pack_mode", "dialogue"))
    assistant_loss_only = bool(data_cfg.get("assistant_loss_only", True))
    seq_len = int(cfg["model"].get("max_seq_len", 160))
    train_loader, val_loader = build_dataloaders(
        text,
        tokenizer,
        seq_len=seq_len,
        batch_size=cfg["train"]["batch_size"],
        val_ratio=data_cfg.get("val_ratio", 0.05),
        num_workers=nw,
        pin_memory=pin,
        val_text=val_text,
        split_seed=int(data_cfg.get("split_seed", seed)),
        pack_mode=pack_mode,
        assistant_loss_only=assistant_loss_only,
    )
    print(
        f"[data] num_workers={nw} pin_memory={pin} "
        f"pack_mode={pack_mode} assistant_loss_only={assistant_loss_only}"
    )

    model = build_transformer_lm(cfg, tokenizer.vocab_size).to(device)
    n_params = model.count_params()
    print(f"params = {n_params:,}  arch=transformer  layers={model.n_layers} d_model={model.d_model}")

    base_lr = float(cfg["train"]["lr"])
    min_lr = float(cfg["train"].get("min_lr", base_lr * 0.1))
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=float(cfg["train"].get("weight_decay", 0.01)),
    )

    out_dir = Path(cfg["train"]["out_dir"])
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_config.json").write_text(
        json.dumps(model.config_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    steps = int(cfg["train"]["steps"])
    total_target = int(cfg["train"].get("total_target") or steps)
    log_every = int(cfg["train"].get("log_every", 50))
    save_every = int(cfg["train"].get("save_every", 500))
    sample_every = int(cfg["train"].get("sample_every", 500))
    eval_every = int(cfg["train"].get("eval_every", save_every or log_every))
    eval_batches = int(cfg["train"].get("eval_batches", 20))
    it = iter(train_loader)
    model.train()
    pbar = tqdm(range(steps), desc="train@transformer")
    running = 0.0
    running_count = 0
    best_val_nll = float("inf")
    last_val_nll = None
    samples_path = out_dir / "samples.txt"
    t0 = time.time()

    def _save(path: Path, global_step: int) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "config": model.config_dict(),
                "step": global_step,
                "optimizer": opt.state_dict(),
                "best_val_nll": best_val_nll,
            },
            path,
        )

    def _evaluate_and_select(global_step: int) -> float:
        nonlocal best_val_nll, last_val_nll
        val_nll = eval_nll(model, val_loader, device, max_batches=eval_batches)
        last_val_nll = val_nll
        model.train()
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            _save(out_dir / "best.pt", global_step)
        return val_nll

    for step in pbar:
        global_step = step + 1
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        inputs, targets = unpack_lm_batch(batch)
        inputs = inputs.to(device, non_blocking=bool(pin))
        targets = targets.to(device, non_blocking=bool(pin))
        lr = _cosine_lr(global_step - 1, total_target, base_lr, min_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        _, loss = model(inputs, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at step {global_step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_f = float(loss.detach())
        running += loss_f
        running_count += 1
        if global_step % log_every == 0:
            avg = running / max(1, running_count)
            pbar.set_postfix(loss=round(avg, 3), lr=f"{lr:.2e}", gstep=global_step)
            running = 0.0
            running_count = 0

        if eval_every > 0 and global_step % eval_every == 0:
            val_nll = _evaluate_and_select(global_step)
            pbar.set_postfix(val_nll=round(val_nll, 3), gstep=global_step)

        if save_every > 0 and global_step % save_every == 0:
            _save(out_dir / f"step_{global_step}.pt", global_step)

        if sample_every > 0 and global_step % sample_every == 0:
            try:
                sample = _sample_chat(model, tokenizer, device)
                with samples_path.open("a", encoding="utf-8") as sf:
                    sf.write(f"\n=== step {global_step} loss={loss_f:.3f} ===\n{sample}\n")
                print("\n" + sample[:200].replace("\n", " / ") + "\n")
            except Exception as e:
                print(f"[sample skipped] {e}")

    final_step = steps
    if eval_every <= 0 or final_step % eval_every != 0:
        _evaluate_and_select(final_step)
    ckpt = out_dir / "checkpoint.pt"
    _save(ckpt, final_step)
    elapsed = time.time() - t0

    task_metrics: dict = {}
    if val_text and bool(cfg["train"].get("eval_compositional", True)):
        composition_items, scramble_items = parse_compositional_tasks(text_blocks(val_text))
        if composition_items or scramble_items:
            kw, scramble = eval_task_metrics(
                model,
                tokenizer,
                composition_items,
                scramble_items,
                device=device,
                max_composition=int(cfg["train"].get("max_composition_eval", 48)),
                max_scramble=int(cfg["train"].get("max_scramble_eval", 48)),
            )
            task_metrics = {
                "keyword_coverage": kw,
                "scramble_exact_match": scramble,
                "n_composition_eval": min(
                    len(composition_items),
                    int(cfg["train"].get("max_composition_eval", 48)),
                ),
                "n_scramble_eval": min(
                    len(scramble_items),
                    int(cfg["train"].get("max_scramble_eval", 48)),
                ),
            }
            print(f"[tasks] keyword_coverage={kw:.3f} scramble_exact_match={scramble:.3f}")

    meta = {
        "arch": "transformer_lm",
        "params": n_params,
        "device": str(device),
        "backend": device_info.backend,
        "best_val_nll": best_val_nll,
        "last_val_nll": last_val_nll,
        "steps": steps,
        "final_step": final_step,
        "elapsed_sec": elapsed,
        "pack_mode": pack_mode,
        "assistant_loss_only": assistant_loss_only,
        "compare_to": "runs/train_modern_moe_attn_v1",
        **task_metrics,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] elapsed={elapsed/60:.1f} min  best_val_nll={best_val_nll:.4f}  params={n_params:,}")
    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Transformer LM baseline")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs" / "train_7m_modern.yaml"),
    )
    args = parser.parse_args()
    ckpt = train_from_config(args.config)
    print("checkpoint:", ckpt)


if __name__ == "__main__":
    main()
