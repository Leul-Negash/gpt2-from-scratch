"""A decoder-only Transformer in the GPT-2 layout, built from tensor operations and nn.Linear / nn.Embedding.

Shapes: B = batch, T = sequence length, C = n_embd, H = n_head, D = C / H (head size), V = vocab size.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Normalise each token's vector to mean 0 and variance 1, then rescale with a learned gain and shift."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias


def gelu(x):
    """GPT-2's tanh approximation of GELU (the original Transformer used ReLU)."""
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def sinusoidal_positions(block_size, dim):
    """PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d)), Vaswani et al. Section 3.5."""
    pos = torch.arange(block_size).unsqueeze(1)
    freq = torch.exp(-math.log(10000.0) * torch.arange(0, dim, 2) / dim)
    pe = torch.zeros(block_size, dim)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe


class CausalSelfAttention(nn.Module):
    """Multi-head scaled dot-product attention: softmax(Q K^T / sqrt(D) + mask) V, run in H heads at once."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head, self.causal = cfg.n_head, cfg.causal
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)   # W_Q, W_K, W_V of every head in one matrix
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)      # W_O, mixes the concatenated heads
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool)),
                             persistent=False)
        self.weights = None

    def forward(self, x):
        B, T, C = x.shape
        H, D = self.n_head, C // self.n_head
        q, k, v = self.qkv(x).split(C, dim=2)                   # each (B, T, C)
        q = q.view(B, T, H, D).transpose(1, 2)                  # (B, H, T, D)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(D)         # (B, H, T, T): how much token i looks at token j
        if self.causal:
            scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))  # token i cannot see j > i
        weights = F.softmax(scores, dim=-1)                     # each row sums to 1
        self.weights = weights.detach()
        out = self.attn_drop(weights) @ v                       # (B, H, T, D): weighted sum of values
        out = out.transpose(1, 2).contiguous().view(B, T, C)    # concatenate the heads back to (B, T, C)
        return self.resid_drop(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise network: the same two-layer MLP applied to every token separately, C -> 4C -> C."""

    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(gelu(self.fc(x))))


class Block(nn.Module):
    """GPT-2 block with pre-LayerNorm: x + Attn(LN(x)), then x + FFN(LN(x)).

    The original Transformer normalised after the residual addition (post-LN): LN(x + Sublayer(x)).
    """

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.n_embd)
        self.ffn = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # tokens exchange information
        x = x + self.ffn(self.ln2(x))    # each token processes what it gathered
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        if cfg.position == "learned":
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        elif cfg.position == "sinusoidal":
            self.register_buffer("pos_table", sinusoidal_positions(cfg.block_size, cfg.n_embd), persistent=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd)                     # GPT-2 adds a final LayerNorm after the last block
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight             # weight tying, as in GPT-2 and Vaswani et al. 3.4
        self.apply(self._init)
        for name, p in self.named_parameters():
            # GPT-2: residual projections start smaller, by 1/sqrt(2 * n_layer), so the stream does not grow with depth
            if name.endswith("proj.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def positions(self, T):
        if self.cfg.position == "learned":
            return self.pos_emb(torch.arange(T, device=self.lm_head.weight.device))
        if self.cfg.position == "sinusoidal":
            return self.pos_table[:T]
        return 0

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence of {T} tokens is longer than the context of {self.cfg.block_size}"
        x = self.drop(self.tok_emb(idx) + self.positions(T))  # (B, T, C): what the token is + where it is
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))                   # (B, T, V): a score for every possible next token
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, generator=None):
        """Autoregressive decoding: predict, append, repeat. temperature = 0 means greedy (always the top token)."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size:])   # the model only sees its last block_size tokens
            logits = logits[:, -1, :]
            if temperature == 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, top_k).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
