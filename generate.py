"""Generate text from a trained checkpoint, greedy or with temperature sampling."""
import argparse
import os

import torch

from data import load_tokenizer
from train import load_model

ROOT = os.path.dirname(os.path.abspath(__file__))


def complete(model, tok, prompt, tokens, temperature, top_k=None, seed=0):
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    out = model.generate(idx, tokens, temperature=temperature, top_k=top_k, generator=g)
    return tok.decode(out[0].tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=os.path.join(ROOT, "checkpoints", "gpt_best.pt"))
    p.add_argument("--prompt", default="ROMEO:\n")
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--greedy", action="store_true", help="always take the most likely token (same as temperature 0)")
    p.add_argument("--top_k", type=int)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    torch.set_num_threads(4)
    model, ckpt = load_model(a.checkpoint)
    tok = load_tokenizer(model.cfg.vocab_size)
    temperature = 0.0 if a.greedy else a.temperature
    print(f"[checkpoint step {ckpt['step']}, val loss {ckpt['history'][-1]['val']:.4f}, "
          f"{'greedy' if temperature == 0 else f'temperature {temperature}'}]")
    print(complete(model, tok, a.prompt.replace("\\n", "\n"), a.tokens, temperature, a.top_k, a.seed))


if __name__ == "__main__":
    main()
