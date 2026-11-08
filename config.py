"""Model and training settings. Sizes are powers of two so tensors line up with memory and SIMD widths."""
from dataclasses import dataclass, asdict


@dataclass
class GPTConfig:
    vocab_size: int = 512      # 256 bytes + 256 BPE merges; GPT-2 has 50,257 but trains on 40 GB, we have 1 MB
    block_size: int = 128      # context length in tokens (GPT-2: 1024); about 250 characters of Shakespeare
    n_layer: int = 4           # Transformer blocks (GPT-2 small: 12)
    n_head: int = 4            # attention heads, head size = n_embd / n_head = 32 (GPT-2: 12 heads of 64)
    n_embd: int = 128          # width of the residual stream (GPT-2 small: 768)
    dropout: float = 0.1       # GPT-2 used 0.1; a small dataset overfits without it
    position: str = "learned"  # "learned" like GPT-2, "sinusoidal" like the original Transformer, or "none"
    causal: bool = True        # the causal mask; switched off only to show what goes wrong without it

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    batch_size: int = 32
    max_steps: int = 5000
    lr: float = 1e-3           # peak learning rate; small models tolerate a larger rate than GPT-2's 2.5e-4
    min_lr: float = 1e-4       # cosine decay ends at lr / 10
    warmup_steps: int = 200    # linear warm-up, so Adam's early noisy steps do not wreck the initial weights
    weight_decay: float = 0.1  # on weight matrices and embeddings only, not on biases or LayerNorm gains
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_batches: int = 20     # 20 x 4,096 tokens, more than the whole validation split
    seed: int = 1337
    threads: int = 4           # leave half the CPU free

    def to_dict(self):
        return asdict(self)
