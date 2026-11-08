"""Tiny Shakespeare: download, tokenize, split, and cut into (input, target) sequences."""
import os
import urllib.request

import torch

from tokenizer import BPETokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
TEXT = os.path.join(DATA, "tinyshakespeare.txt")
TOKENIZER = os.path.join(ROOT, "tokenizer.json")


def load_text():
    os.makedirs(DATA, exist_ok=True)
    if not os.path.exists(TEXT):
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, TEXT)
    with open(TEXT, encoding="utf-8") as f:
        return f.read()


def load_tokenizer(vocab_size=512):
    if os.path.exists(TOKENIZER):
        return BPETokenizer.load(TOKENIZER)
    text = load_text()
    # the tokenizer only sees the training part, so validation text stays unseen
    tok = BPETokenizer.train(text[: int(0.9 * len(text))], vocab_size, verbose=True)
    tok.save(TOKENIZER)
    return tok


def load_splits(vocab_size=512):
    """First 90% of the play text for training, last 10% for validation (a contiguous split, so no line leaks)."""
    text = load_text()
    tok = load_tokenizer(vocab_size)
    cut = int(0.9 * len(text))
    train = torch.tensor(tok.encode(text[:cut]), dtype=torch.long)
    val = torch.tensor(tok.encode(text[cut:]), dtype=torch.long)
    return tok, train, val


def get_batch(ids, batch_size, block_size, generator=None):
    """Random windows of block_size + 1 tokens; the target is the input shifted left by one."""
    start = torch.randint(len(ids) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([ids[s: s + block_size] for s in start])
    y = torch.stack([ids[s + 1: s + block_size + 1] for s in start])
    return x, y
