"""Training loop for a fixed (usually evolved) genome."""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from evo_lm.arch.hybrid import Genome, build_model
from evo_lm.data.dataset import build_dataloaders, load_text_files, unpack_lm_batch
from evo_lm.data.tokenizer import CharTokenizer
from evo_lm.eval.fitness import eval_nll, eval_task_metrics, parse_compositional_tasks
from evo_lm.runtime.device import detect_device
from evo_lm.runtime.vulkan_ops import export_for_vulkan


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

    prefer = cfg.get("device", os.environ.get("EVO_DEVICE", "vulkan"))
    if cfg.get("allow_cpu"):
        os.environ["EVO_ALLOW_CPU"] = "1"
    device_info = detect_device(prefer)
    device = device_info.torch_device
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    print(device_info)
    if device_info.backend == "cpu" and not cfg.get("allow_cpu"):
        raise RuntimeError("Refusing CPU-only train; set device: rocm|vulkan or allow_cpu: true.")

    use_vulkan = bool(cfg.get("use_vulkan", device_info.backend == "vulkan"))
    if device_info.backend == "rocm":
        use_vulkan = False
        print("[train] ROCm/HIP active → full model on GPU (no Vulkan Linear path).")
    # Long speech runs: native matmul is far faster than sync-heavy Vulkan GEMM.
    if cfg.get("fast_host_matmul"):
        use_vulkan = False
        print("[train] fast_host_matmul=true → Linear uses host BLAS (Zen cores); Vulkan export still available.")

    data_cfg = cfg["data"]
    paths = [Path(p) for p in data_cfg["texts"]]
    text = load_text_files(paths)
    if not text.strip():
        raise FileNotFoundError(f"No text loaded from {paths}. Run prepare / build_speak_data first.")

    tok_path = Path(data_cfg.get("tokenizer", "data/processed/tokenizer.json"))
    rebuild_tok = bool(data_cfg.get("rebuild_tokenizer", False))
    if tok_path.exists() and not rebuild_tok:
        tokenizer = CharTokenizer.load(tok_path)
    else:
        tokenizer = CharTokenizer.train_from_texts([text], max_vocab=data_cfg.get("max_vocab", 12000))
        tok_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(tok_path)

    # Prefetch on CPU while GPU computes — better than splitting the model across devices.
    nw = int(cfg["train"].get("num_workers", 4 if device_info.backend == "rocm" else 0))
    pin = bool(cfg["train"].get("pin_memory", device_info.backend == "rocm"))
    val_paths = [Path(p) for p in data_cfg.get("val_texts", [])]
    val_text = load_text_files(val_paths) if val_paths else None
    pack_mode = str(data_cfg.get("pack_mode", "dialogue"))
    assistant_loss_only = bool(data_cfg.get("assistant_loss_only", True))
    train_loader, val_loader = build_dataloaders(
        text,
        tokenizer,
        seq_len=cfg["model"].get("max_seq_len", 256),
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

    resume_path = cfg["train"].get("resume") or cfg["model"].get("resume")
    start_step = 0
    resumed = None
    if resume_path and Path(resume_path).exists():
        resumed = torch.load(resume_path, map_location="cpu", weights_only=False)
        genome = Genome.from_dict(resumed["genome"])
        genome.vocab_size = tokenizer.vocab_size
        start_step = int(resumed.get("step") or 0)
        print(f"[resume] {resume_path} @ step {start_step}")
    else:
        genome_path = cfg["model"].get("genome")
        if genome_path and Path(genome_path).exists():
            genome = Genome.load(genome_path)
            genome.vocab_size = tokenizer.vocab_size
            if cfg["model"].get("max_seq_len"):
                genome.max_seq_len = int(cfg["model"]["max_seq_len"])
        else:
            from evo_lm.arch.hybrid import BlockGene

            blocks = []
            pattern = cfg["model"].get("block_pattern", ["ssm", "gru", "ssm", "conv", "ssm", "mlp"])
            for kind in pattern:
                blocks.append(BlockGene(kind=kind, expand=int(cfg["model"].get("expand", 2))))
            genome = Genome(
                d_model=cfg["model"]["d_model"],
                vocab_size=tokenizer.vocab_size,
                max_seq_len=cfg["model"].get("max_seq_len", 256),
                blocks=blocks,
            )

    model = build_model(genome, device=device, use_vulkan=use_vulkan)
    if resumed is not None:
        missing, unexpected = model.load_state_dict(resumed["model"], strict=False)
        if missing:
            print(f"[resume] missing keys: {len(missing)}")
        if unexpected:
            print(f"[resume] unexpected keys: {len(unexpected)}")
    print(f"params = {model.count_params():,}  threads={threads}  vulkan_linear={use_vulkan}")

    base_lr = float(cfg["train"]["lr"])
    min_lr = float(cfg["train"].get("min_lr", base_lr * 0.1))
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=float(cfg["train"].get("weight_decay", 0.01)))
    if resumed is not None and resumed.get("optimizer"):
        try:
            opt.load_state_dict(resumed["optimizer"])
            print("[resume] optimizer state restored")
        except Exception as e:
            print(f"[resume] optimizer not restored ({e})")

    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    genome.save(str(out_dir / "genome.json"))

    # steps = additional steps from now; total_target optional for LR schedule
    steps = int(cfg["train"]["steps"])
    total_target = int(cfg["train"].get("total_target") or (start_step + steps))
    log_every = int(cfg["train"].get("log_every", 50))
    save_every = int(cfg["train"].get("save_every", 500))
    sample_every = int(cfg["train"].get("sample_every", 500))
    eval_every = int(cfg["train"].get("eval_every", save_every or log_every))
    eval_batches = int(cfg["train"].get("eval_batches", 20))
    moe_aux_coef = float(cfg["train"].get("moe_aux_coef", 0.01))
    it = iter(train_loader)
    model.train()
    pbar = tqdm(range(steps), desc=f"train@{start_step}")
    running = 0.0
    running_count = 0
    best_val_nll = (
        float(resumed.get("best_val_nll", float("inf")))
        if resumed is not None
        else float("inf")
    )
    last_val_nll = None
    samples_path = out_dir / "samples.txt"

    def _save(path: Path, global_step: int) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "genome": genome.to_dict(),
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
        global_step = start_step + step + 1
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
        objective = loss + moe_aux_coef * model.moe_aux_loss
        if not torch.isfinite(objective):
            raise FloatingPointError(f"Non-finite training objective at step {global_step}")
        opt.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_f = float(loss.detach())
        aux_f = float(model.moe_aux_loss.detach())
        running += loss_f
        running_count += 1
        if global_step % log_every == 0:
            avg = running / max(1, running_count)
            pbar.set_postfix(
                loss=round(avg, 3),
                moe=round(aux_f, 3),
                lr=f"{lr:.2e}",
                gstep=global_step,
            )
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

    final_step = start_step + steps
    if eval_every <= 0 or final_step % eval_every != 0:
        _evaluate_and_select(final_step)
    ckpt = out_dir / "checkpoint.pt"
    _save(ckpt, final_step)

    # Optional compositional probe on validation dialogue tasks.
    task_metrics = {}
    if val_text and bool(cfg["train"].get("eval_compositional", True)):
        from evo_lm.data.dataset import text_blocks

        composition_items, scramble_items = parse_compositional_tasks(text_blocks(val_text))
        if composition_items or scramble_items:
            kw, scramble = eval_task_metrics(
                model,
                tokenizer,
                composition_items,
                scramble_items,
                device=device,
                max_composition=int(cfg["train"].get("max_composition_eval", 32)),
                max_scramble=int(cfg["train"].get("max_scramble_eval", 32)),
            )
            task_metrics = {
                "keyword_coverage": kw,
                "scramble_exact_match": scramble,
                "n_composition_eval": min(len(composition_items), int(cfg["train"].get("max_composition_eval", 32))),
                "n_scramble_eval": min(len(scramble_items), int(cfg["train"].get("max_scramble_eval", 32))),
            }
            print(
                f"[tasks] keyword_coverage={kw:.3f} "
                f"scramble_exact_match={scramble:.3f}"
            )

    if cfg["train"].get("export_vulkan", True):
        # export uses plain modules; rebuild without vulkan wrappers if needed
        export_model = model
        if use_vulkan:
            export_model = build_model(genome, device="cpu", use_vulkan=False)
            export_model.load_state_dict(
                {k: v.cpu() for k, v in model.state_dict().items()},
                strict=False,
            )
        export_for_vulkan(export_model, out_dir / "vulkan_export")
    meta = {
        "params": model.count_params(),
        "device": str(device),
        "backend": device_info.backend,
        "use_vulkan_linear": use_vulkan,
        "best_loss": best_val_nll,
        "best_val_nll": best_val_nll,
        "last_val_nll": last_val_nll,
        "moe_aux_coef": moe_aux_coef,
        "steps": steps,
        "start_step": start_step,
        "final_step": final_step,
        "pack_mode": pack_mode,
        "assistant_loss_only": assistant_loss_only,
        **task_metrics,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return ckpt
