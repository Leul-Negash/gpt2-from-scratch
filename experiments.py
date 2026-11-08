"""Experiments: temperature, prompt length and context, and architecture (heads, layers, positions, causal mask)."""
import argparse
import json
import os
import re
import time

import torch
import torch.nn.functional as F

from config import GPTConfig, TrainConfig
from data import load_splits, load_text
from generate import complete
from train import CHECKPOINTS, RESULTS, fixed_batches, load_model, train

MAIN = os.path.join(CHECKPOINTS, "gpt_best.pt")
EXP_STEPS = 800


def words(text):
    return re.findall(r"[a-z']+", text.lower())


def text_stats(sample, known):
    w = words(sample)
    grams = list(zip(w, w[1:], w[2:], w[3:]))
    return {"real_words": sum(x in known for x in w) / max(1, len(w)),
            "repeated_4grams": 1 - len(set(grams)) / max(1, len(grams)),
            "distinct_words": len(set(w)) / max(1, len(w))}


def save(name, obj):
    path = os.path.join(RESULTS, f"{name}.json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


@torch.no_grad()
def temperature_experiment():
    model, _ = load_model(MAIN)
    tok, train_ids, val_ids = load_splits()
    text = load_text()
    known = set(words(text[: int(0.9 * len(text))]))
    batches = fixed_batches(val_ids, 10, 32, model.cfg.block_size, seed=2)
    temps = [0.0, 0.5, 0.8, 1.0, 1.5, 2.0]
    out = {}
    print("=" * 78)
    print("A. Temperature: how sharp the next-token distribution is (validation set)")
    print("=" * 78)
    print(f"  {'T':<6}{'entropy (nats)':>16}{'top-1 prob':>12}{'val loss':>11}")
    for T in temps[1:]:
        ent, top, loss = 0.0, 0.0, 0.0
        for x, y in batches:
            logits = model(x)[0] / T
            logp = F.log_softmax(logits, dim=-1)
            ent += -(logp.exp() * logp).sum(-1).mean().item()
            top += logp.exp().max(-1).values.mean().item()
            loss += F.nll_loss(logp.view(-1, logp.size(-1)), y.view(-1)).item()
        n = len(batches)
        out[T] = {"entropy": ent / n, "top1": top / n, "val_loss": loss / n}
        print(f"  {T:<6}{ent / n:>16.3f}{top / n:>12.3f}{loss / n:>11.4f}")
    print("\n" + "=" * 78)
    print("B. Temperature: 8 samples of 150 tokens each from the prompt 'ROMEO:'")
    print("=" * 78)
    print(f"  {'T':<9}{'real words':>12}{'repeated 4-grams':>18}{'distinct words':>16}")
    samples = {}
    for T in temps:
        texts = [complete(model, tok, "ROMEO:\n", 150, T, seed=s) for s in range(1 if T == 0 else 8)]
        stats = [text_stats(t, known) for t in texts]
        avg = {k: sum(s[k] for s in stats) / len(stats) for k in stats[0]}
        out.setdefault(T, {}).update(avg)
        samples[T] = texts[0]
        label = "greedy" if T == 0 else T
        print(f"  {label:<9}{avg['real_words']:>12.1%}{avg['repeated_4grams']:>18.1%}{avg['distinct_words']:>16.1%}")
    print()
    for T, s in samples.items():
        print(f"--- {'greedy' if T == 0 else f'temperature {T}'} ---")
        print(s.strip()[:420])
        print()
    save("temperature", {"metrics": out, "samples": samples})


@torch.no_grad()
def context_experiment():
    model, _ = load_model(MAIN)
    tok, _, val_ids = load_splits()
    B = model.cfg.block_size
    # the same 2,048 validation tokens are predicted every time; only the amount of context before them changes
    g = torch.Generator().manual_seed(3)
    targets = torch.randint(B, len(val_ids), (2048,), generator=g)
    lengths = [1, 2, 4, 8, 16, 32, 64, 128]
    per_len = {}
    for n in lengths:
        loss = 0.0
        for chunk in targets.split(256):
            x = torch.stack([val_ids[p - n: p] for p in chunk])
            loss += F.cross_entropy(model(x)[0][:, -1], val_ids[chunk], reduction="sum").item()
        per_len[n] = loss / len(targets)
    print("=" * 78)
    print("C. Context: loss on the same 2,048 validation tokens, predicted from the previous n tokens only")
    print("=" * 78)
    print(f"  {'n tokens seen':<15}{'loss':>8}{'perplexity':>12}")
    for n in lengths:
        print(f"  {n:<15}{per_len[n]:>8.4f}{torch.tensor(per_len[n]).exp().item():>12.1f}")

    passage = tok.decode(val_ids[3000:3000 + 200].tolist())
    start = passage.index("\n\n") + 2
    ids = tok.encode(passage[start:])
    prompts = {"1 token": tok.decode(ids[:1]), "16 tokens": tok.decode(ids[:16]), "112 tokens": tok.decode(ids[:112])}
    print("\n" + "=" * 78)
    print("D. Prompt length: greedy continuation (80 tokens) of the same validation scene")
    print("=" * 78)
    samples = {}
    for label, prompt in prompts.items():
        full = complete(model, tok, prompt, 80, 0.0)
        samples[label] = {"prompt": prompt, "continuation": full[len(prompt):]}
        print(f"--- prompt: {label} ---")
        print(f"[prompt]{prompt[-160:]!r}")
        print(f"[model] {full[len(prompt):].strip()[:300]}")
        print()
    save("context", {"loss_by_context": per_len, "samples": samples})


@torch.no_grad()
def last_position_loss(model, val_ids):
    """Loss of the final position only: the one place where no future token exists to peek at."""
    loss = 0.0
    batches = fixed_batches(val_ids, 20, 32, model.cfg.block_size, seed=2)
    for x, y in batches:
        loss += F.cross_entropy(model(x)[0][:, -1], y[:, -1]).item()
    return loss / len(batches)


VARIANTS = {
    "base": {},
    "heads_1": {"n_head": 1}, "heads_2": {"n_head": 2}, "heads_8": {"n_head": 8},
    "layers_1": {"n_layer": 1}, "layers_2": {"n_layer": 2}, "layers_6": {"n_layer": 6},
    "pos_sinusoidal": {"position": "sinusoidal"}, "pos_none": {"position": "none"},
    "no_mask": {"causal": False},
}


def architecture_experiment(names, rerun=False):
    _, _, val_ids = load_splits()
    rows = {}
    for name in names:
        path = os.path.join(RESULTS, f"exp_{name}_history.json")
        if os.path.exists(path) and not rerun:
            with open(path) as f:
                summary = json.load(f)
        else:
            tick = time.perf_counter()
            summary = train(GPTConfig(**VARIANTS[name]), TrainConfig(max_steps=EXP_STEPS), f"exp_{name}")
            summary["seconds_per_step"] = (time.perf_counter() - tick) / EXP_STEPS
            model, _ = load_model(os.path.join(CHECKPOINTS, f"exp_{name}_last.pt"))
            summary["last_position_val"] = last_position_loss(model, val_ids)
            with open(path, "w") as f:
                json.dump(summary, f, indent=2)
        rows[name] = summary
    print("=" * 78)
    print(f"E. Architecture: every model trained {EXP_STEPS} steps with the same data, seed and schedule")
    print("=" * 78)
    print(f"  {'variant':<16}{'params':>9}{'train':>9}{'val':>9}{'last-pos val':>14}{'s/step':>9}")
    for name, s in rows.items():
        h = s["history"][-1]
        print(f"  {name:<16}{s['params']:>9,}{h['train']:>9.4f}{h['val']:>9.4f}"
              f"{s['last_position_val']:>14.4f}{s['seconds_per_step']:>9.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("which", nargs="*", default=["temperature", "context", "architecture"])
    p.add_argument("--variants", nargs="*", default=list(VARIANTS))
    p.add_argument("--rerun", action="store_true", help="retrain architecture variants even if results exist")
    a = p.parse_args()
    torch.set_num_threads(4)
    for which in a.which:
        if which == "temperature":
            temperature_experiment()
        elif which == "context":
            context_experiment()
        elif which == "architecture":
            architecture_experiment(a.variants, a.rerun)
        print()


if __name__ == "__main__":
    main()
