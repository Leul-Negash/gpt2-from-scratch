import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GPTConfig, TrainConfig  # noqa: E402
from model import GPT, CausalSelfAttention, LayerNorm, gelu, sinusoidal_positions  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402

torch.set_num_threads(2)
TINY = dict(vocab_size=300, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.0)
TEXT = "To be, or not to be: that is the question. Whether 'tis nobler in the mind to suffer.\n" * 20


def test_tokenizer_round_trip_and_merges():
    tok = BPETokenizer.train(TEXT, 300)
    ids = tok.encode(TEXT)
    assert tok.decode(ids) == TEXT
    assert len(ids) < len(TEXT.encode("utf-8"))           # merges compress the byte sequence
    assert tok.decode(tok.encode("unseen wörds")) == "unseen wörds"  # any byte string still works


def test_tokenizer_save_load(tmp_path):
    tok = BPETokenizer.train(TEXT, 300)
    tok.save(tmp_path / "tok.json")
    assert BPETokenizer.load(tmp_path / "tok.json").encode(TEXT) == tok.encode(TEXT)


def test_layernorm_matches_pytorch():
    x = torch.randn(3, 5, 32)
    ln = LayerNorm(32)
    with torch.no_grad():
        ln.weight.uniform_(0.5, 1.5)
        ln.bias.uniform_(-0.5, 0.5)
    assert torch.allclose(ln(x), F.layer_norm(x, (32,), ln.weight, ln.bias, 1e-5), atol=1e-5)


def test_gelu_matches_pytorch():
    x = torch.linspace(-6, 6, 101)
    assert torch.allclose(gelu(x), F.gelu(x, approximate="tanh"), atol=1e-6)


def test_attention_matches_reference():
    """My attention gives the same result as PyTorch's fused kernel, used here only as a reference."""
    cfg = GPTConfig(**TINY)
    attn = CausalSelfAttention(cfg).eval()
    x = torch.randn(2, 10, 32)
    q, k, v = attn.qkv(x).split(32, dim=2)
    q, k, v = (t.view(2, 10, 4, 8).transpose(1, 2) for t in (q, k, v))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(2, 10, 32)
    assert torch.allclose(attn(x), attn.proj(ref), atol=1e-5)


def test_attention_weights_are_causal_and_normalised():
    attn = CausalSelfAttention(GPTConfig(**TINY)).eval()
    attn(torch.randn(2, 10, 32))
    w = attn.weights
    assert w.shape == (2, 4, 10, 10)
    assert torch.allclose(w.sum(-1), torch.ones(2, 4, 10))
    assert torch.all(w.triu(diagonal=1) == 0)


def test_future_tokens_do_not_change_past_predictions():
    model = GPT(GPTConfig(**TINY)).eval()
    a = torch.randint(0, 300, (1, 12))
    b = a.clone()
    b[0, 8:] = torch.randint(0, 300, (4,))
    la, lb = model(a)[0], model(b)[0]
    assert torch.allclose(la[0, :8], lb[0, :8], atol=1e-6)
    assert not torch.allclose(la[0, 8:], lb[0, 8:])


def test_without_mask_future_tokens_leak():
    model = GPT(GPTConfig(**TINY, causal=False)).eval()
    a = torch.randint(0, 300, (1, 12))
    b = a.clone()
    b[0, -1] = (a[0, -1] + 1) % 300
    assert not torch.allclose(model(a)[0][0, 0], model(b)[0][0, 0])


def test_shapes_and_initial_loss():
    model = GPT(GPTConfig(**TINY))
    x, y = torch.randint(0, 300, (4, 16)), torch.randint(0, 300, (4, 16))
    logits, loss = model(x, y)
    assert logits.shape == (4, 16, 300)
    assert abs(loss.item() - math.log(300)) < 0.1          # untrained model is close to uniform over the vocab


def test_weight_tying():
    model = GPT(GPTConfig(**TINY))
    assert model.lm_head.weight is model.tok_emb.weight


def test_sinusoidal_positions_formula():
    pe = sinusoidal_positions(16, 32)
    pos, i = 5, 3
    assert torch.isclose(pe[pos, 2 * i], torch.tensor(math.sin(pos / 10000 ** (2 * i / 32))))
    assert torch.isclose(pe[pos, 2 * i + 1], torch.tensor(math.cos(pos / 10000 ** (2 * i / 32))))


@pytest.mark.parametrize("position", ["learned", "sinusoidal", "none"])
def test_model_overfits_one_batch(position):
    torch.manual_seed(0)
    model = GPT(GPTConfig(**TINY, position=position))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, 300, (4, 16))
    y = torch.roll(x, -1, dims=1)
    for _ in range(150):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5


def test_generation_greedy_is_deterministic_and_sampling_respects_context():
    model = GPT(GPTConfig(**TINY)).eval()
    start = torch.zeros(1, 1, dtype=torch.long)
    a = model.generate(start, 30, temperature=0)
    b = model.generate(start, 30, temperature=0)
    assert torch.equal(a, b) and a.shape == (1, 31)       # generates past block_size by cropping the context
    s = model.generate(start, 30, temperature=1.0, top_k=5, generator=torch.Generator().manual_seed(0))
    assert s.shape == (1, 31)


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    import data
    import train

    text = TEXT * 5
    monkeypatch.setattr(data, "load_text", lambda: text)
    monkeypatch.setattr(train, "CHECKPOINTS", str(tmp_path))
    monkeypatch.setattr(train, "RESULTS", str(tmp_path))
    tok = BPETokenizer.train(text[: int(0.9 * len(text))], 300)
    ids = torch.tensor(tok.encode(text))
    monkeypatch.setattr(train, "load_splits", lambda v: (tok, ids[:1500], ids[1500:]))
    cfg = GPTConfig(**{**TINY, "dropout": 0.1})
    t = TrainConfig(max_steps=20, eval_interval=10, eval_batches=2, batch_size=4, warmup_steps=5, threads=2)
    full = train.train(cfg, t, "full")["history"]
    train.train(cfg, t, "split", stop_at=10)
    resumed = train.train(cfg, t, "split", resume=True)["history"]
    assert [h["step"] for h in resumed] == [0, 10, 20]
    assert resumed[-1]["val"] == pytest.approx(full[-1]["val"], abs=1e-6)
    model, _ = train.load_model(os.path.join(tmp_path, "full_best.pt"))
    assert isinstance(model, GPT)
