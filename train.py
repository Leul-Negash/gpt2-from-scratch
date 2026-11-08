"""Train the GPT on Tiny Shakespeare with next-token cross-entropy, tracking train and validation loss."""
import argparse
import json
import math
import os
import time

import torch

from config import GPTConfig, TrainConfig
from data import get_batch, load_splits
from model import GPT

ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS = os.path.join(ROOT, "checkpoints")
RESULTS = os.path.join(ROOT, "results")


def lr_at(step, t):
    """Linear warm-up, then cosine decay from lr to min_lr."""
    if step < t.warmup_steps:
        return t.lr * (step + 1) / t.warmup_steps
    progress = min(1.0, (step - t.warmup_steps) / max(1, t.max_steps - t.warmup_steps))
    return t.min_lr + 0.5 * (t.lr - t.min_lr) * (1 + math.cos(math.pi * progress))


def make_optimizer(model, t):
    """AdamW; weight decay only on matrices (weights, embeddings), not on biases and LayerNorm parameters."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": t.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=t.lr, betas=t.betas)


def fixed_batches(ids, n, batch_size, block_size, seed=0):
    """The same evaluation windows at every evaluation, so the loss curve is not noisy from batch choice."""
    g = torch.Generator().manual_seed(seed)
    return [get_batch(ids, batch_size, block_size, g) for _ in range(n)]


@torch.no_grad()
def evaluate(model, batches):
    model.eval()
    loss = sum(model(x, y)[1].item() for x, y in batches) / len(batches)
    model.train()
    return loss


def save_checkpoint(path, model, optimizer, step, history, cfg, t, generator):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None,
                "step": step, "history": history, "config": cfg.to_dict(), "train_config": t.to_dict(),
                "generator": generator.get_state() if generator else None, "torch_rng": torch.get_rng_state()}, path)


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def train(cfg, t, name, resume=False, stop_at=None):
    torch.set_num_threads(t.threads)
    torch.manual_seed(t.seed)
    tok, train_ids, val_ids = load_splits(cfg.vocab_size)
    model = GPT(cfg)
    optimizer = make_optimizer(model, t)
    g = torch.Generator().manual_seed(t.seed)
    history, start = [], 0
    os.makedirs(CHECKPOINTS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    last_path = os.path.join(CHECKPOINTS, f"{name}_last.pt")
    best_path = os.path.join(CHECKPOINTS, f"{name}_best.pt")
    if resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        g.set_state(ckpt["generator"])
        torch.set_rng_state(ckpt["torch_rng"])  # dropout masks continue exactly where they stopped
        history, start = ckpt["history"], ckpt["step"]
        print(f"resumed {name} from step {start}")

    train_eval = fixed_batches(train_ids, t.eval_batches, t.batch_size, cfg.block_size, seed=1)
    val_eval = fixed_batches(val_ids, t.eval_batches, t.batch_size, cfg.block_size, seed=2)
    best = min((h["val"] for h in history), default=float("inf"))
    print(f"{name}: {model.num_params():,} parameters, {cfg.to_dict()}")
    print(f"{len(train_ids):,} training tokens, {len(val_ids):,} validation tokens, "
          f"{t.batch_size * cfg.block_size:,} tokens per step")
    tick = time.perf_counter()
    for step in range(start, t.max_steps + 1):
        due = step % t.eval_interval == 0 or step == t.max_steps
        if due and not (history and history[-1]["step"] == step):
            done = step - (history[-1]["step"] if history else 0)
            speed = f" | {(time.perf_counter() - tick) / done:.2f} s/step" if done else ""
            tr, va = evaluate(model, train_eval), evaluate(model, val_eval)
            history.append({"step": step, "train": tr, "val": va, "lr": lr_at(step, t)})
            print(f"step {step:>5} | train {tr:.4f} | val {va:.4f} | lr {lr_at(step, t):.2e}{speed}", flush=True)
            if va < best:
                best = va
                save_checkpoint(best_path, model, None, step, history, cfg, t, None)
            save_checkpoint(last_path, model, optimizer, step, history, cfg, t, g)
            tick = time.perf_counter()
        if step == t.max_steps or step == stop_at:
            break
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, t)
        x, y = get_batch(train_ids, t.batch_size, cfg.block_size, g)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
        optimizer.step()

    summary = {"name": name, "params": model.num_params(), "config": cfg.to_dict(), "train_config": t.to_dict(),
               "best_val": best, "history": history}
    with open(os.path.join(RESULTS, f"{name}_history.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="gpt")
    p.add_argument("--steps", type=int, default=TrainConfig.max_steps)
    p.add_argument("--n_layer", type=int, default=GPTConfig.n_layer)
    p.add_argument("--n_head", type=int, default=GPTConfig.n_head)
    p.add_argument("--n_embd", type=int, default=GPTConfig.n_embd)
    p.add_argument("--block_size", type=int, default=GPTConfig.block_size)
    p.add_argument("--dropout", type=float, default=GPTConfig.dropout)
    p.add_argument("--position", default=GPTConfig.position, choices=["learned", "sinusoidal", "none"])
    p.add_argument("--no_causal", action="store_true", help="remove the causal mask (to show why it is needed)")
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop_at", type=int, help="pause at this step (it should be an evaluation step); continue with --resume")
    a = p.parse_args()
    cfg = GPTConfig(n_layer=a.n_layer, n_head=a.n_head, n_embd=a.n_embd, block_size=a.block_size,
                    dropout=a.dropout, position=a.position, causal=not a.no_causal)
    t = TrainConfig(max_steps=a.steps, lr=a.lr, min_lr=a.lr / 10)
    train(cfg, t, a.name, resume=a.resume, stop_at=a.stop_at)


if __name__ == "__main__":
    main()
