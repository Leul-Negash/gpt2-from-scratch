"""Figures for the README and the report, made from the saved results and the trained checkpoint."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from data import load_tokenizer  # noqa: E402
from train import CHECKPOINTS, RESULTS, load_model  # noqa: E402

FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
INDIGO, ORANGE, GREEN, GREY, RED = "#4f46e5", "#ea580c", "#059669", "#6b7280", "#dc2626"


def load(name):
    with open(os.path.join(RESULTS, f"{name}.json")) as f:
        return json.load(f)


def loss_curves():
    h = load("gpt_history")["history"]
    steps = [r["step"] for r in h]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(steps, [r["train"] for r in h], color=INDIGO, label="train")
    ax.plot(steps, [r["val"] for r in h], color=ORANGE, label="validation")
    best = min(h, key=lambda r: r["val"])
    ax.scatter([best["step"]], [best["val"]], color=ORANGE, zorder=3)
    ax.annotate(f"best val {best['val']:.3f} (step {best['step']})", (best["step"], best["val"]),
                textcoords="offset points", xytext=(-150, 22), fontsize=8, color=ORANGE)
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy (nats per token)")
    ax.set_ylim(top=6.4)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "loss_curves.png"), dpi=160)


def architecture():
    groups = [("Attention heads (C = 128)", [("heads_1", "1 head"), ("heads_2", "2 heads"), ("base", "4 heads"),
                                             ("heads_8", "8 heads")]),
              ("Transformer layers", [("layers_1", "1 layer"), ("layers_2", "2 layers"), ("base", "4 layers"),
                                      ("layers_6", "6 layers")]),
              ("Positional information", [("base", "learned (GPT-2)"), ("pos_sinusoidal", "sinusoidal (2017)"),
                                          ("pos_none", "none")])]
    colors = [[GREY, GREEN, INDIGO, ORANGE], [GREY, GREEN, INDIGO, ORANGE], [INDIGO, GREEN, GREY]]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    for ax, (title, runs), group_colors in zip(axes, groups, colors):
        for (name, label), c in zip(runs, group_colors):
            h = load(f"exp_{name}_history")["history"]
            ax.plot([r["step"] for r in h], [r["val"] for r in h], color=c, label=label)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step")
        ax.set_ylim(3.3, 5.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("validation loss")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "architecture.png"), dpi=160)


def context():
    by_len = load("context")["loss_by_context"]
    n = [int(k) for k in by_len]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(n, list(by_len.values()), "o-", color=INDIGO)
    ax.set_xscale("log", base=2)
    ax.set_xticks(n, [str(k) for k in n])
    ax.set_xlabel("tokens of context before the predicted token")
    ax.set_ylabel("validation loss (same 2,048 tokens)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "context.png"), dpi=160)


def temperature():
    m = load("temperature")["metrics"]
    temps = sorted(float(t) for t in m)
    sampled = [t for t in temps if t > 0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    a1.plot(sampled, [m[str(t)]["entropy"] for t in sampled], "o-", color=INDIGO, label="entropy (nats)")
    a1.plot(sampled, [m[str(t)]["val_loss"] for t in sampled], "s-", color=ORANGE, label="val loss")
    a1.set_xlabel("temperature")
    a1.set_title("next-token distribution", fontsize=10)
    a1.grid(alpha=0.3)
    a1.legend(fontsize=8)
    a2.plot(temps, [m[str(t)]["real_words"] for t in temps], "o-", color=GREEN, label="real words")
    a2.plot(temps, [m[str(t)]["repeated_4grams"] for t in temps], "s-", color=RED, label="repeated 4-grams")
    a2.set_xlabel("temperature (0 = greedy)")
    a2.set_title("generated text", fontsize=10)
    a2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a2.grid(alpha=0.3)
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "temperature.png"), dpi=160)


@torch.no_grad()
def attention():
    model, _ = load_model(os.path.join(CHECKPOINTS, "gpt_best.pt"))
    tok = load_tokenizer()
    ids = tok.encode("ROMEO:\nO, she doth teach the torches to burn bright!")
    model(torch.tensor([ids]))
    labels = [tok.decode([i]).replace("\n", "\\n") for i in ids]
    L, H = model.cfg.n_layer, model.cfg.n_head
    fig, axes = plt.subplots(L, H, figsize=(2.6 * H, 2.6 * L))
    for l in range(L):
        for h in range(H):
            ax = axes[l][h]
            ax.imshow(model.blocks[l].attn.weights[0, h], cmap="Purples", vmin=0, vmax=1)
            ax.set_xticks(range(len(ids)), labels, rotation=90, fontsize=5)
            ax.set_yticks(range(len(ids)), labels, fontsize=5)
            ax.set_title(f"layer {l + 1}, head {h + 1}", fontsize=8)
    fig.suptitle("Attention weights (row = query token, column = key token it attends to)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "attention.png"), dpi=160)


def diagram():
    fig, ax = plt.subplots(figsize=(6.2, 8.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 17)
    ax.axis("off")

    def box(y, text, color, shape="", w=6.0, x=2.0, h=0.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                                    fc=color, ec="#312e81", lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)
        if shape:
            ax.text(x + w + 0.15, y + h / 2, shape, ha="left", va="center", fontsize=7.5, color=GREY)

    def arrow(y0, y1, x=5.0):
        ax.annotate("", (x, y1), (x, y0), arrowprops=dict(arrowstyle="->", color="#312e81"))

    box(0.3, "token IDs", "#f1f5f9", "(B, T)")
    arrow(1.1, 1.6)
    box(1.6, "token embedding  +  position embedding", "#e0e7ff", "(B, T, C)")
    arrow(2.4, 3.0)
    ax.add_patch(FancyBboxPatch((1.4, 3.0), 7.2, 7.6, boxstyle="round,pad=0.1,rounding_size=0.3",
                                fc="none", ec=INDIGO, lw=1.4, ls="--"))
    ax.text(1.55, 10.3, "Transformer block  × n_layer", fontsize=8.5, color=INDIGO, weight="bold")
    box(3.3, "LayerNorm", "#ecfdf5", w=4.0, x=3.0)
    arrow(4.1, 4.5)
    box(4.5, "masked multi-head self-attention", "#fef3c7", "(B, H, T, T) scores", w=4.0, x=3.0)
    arrow(5.3, 5.8)
    box(5.8, "+  (residual)", "#f1f5f9", w=4.0, x=3.0)
    arrow(6.6, 7.0)
    box(7.0, "LayerNorm", "#ecfdf5", w=4.0, x=3.0)
    arrow(7.8, 8.2)
    box(8.2, "feed-forward  C → 4C → C (GELU)", "#fef3c7", w=4.0, x=3.0)
    arrow(9.0, 9.4)
    box(9.4, "+  (residual)", "#f1f5f9", "(B, T, C)", w=4.0, x=3.0)
    for y0, y1 in [(3.0, 6.2), (6.6, 9.8)]:
        ax.annotate("", (3.0, y1), (2.0, y0 + 0.2), arrowprops=dict(arrowstyle="->", color=GREY,
                                                                   connectionstyle="angle,angleA=90,angleB=0"))
    arrow(10.7, 11.3)
    box(11.3, "final LayerNorm", "#ecfdf5", "(B, T, C)")
    arrow(12.1, 12.6)
    box(12.6, "linear head (tied to token embedding)", "#e0e7ff", "(B, T, V) logits")
    arrow(13.4, 13.9)
    box(13.9, "softmax over the vocabulary → next token", "#f1f5f9", "(B, T, V)")
    ax.text(5.0, 15.3, "GPT-2 style decoder-only Transformer", ha="center", fontsize=11, weight="bold",
            color="#312e81")
    ax.text(5.0, 14.9, "B batch, T tokens, C width, H heads, V vocabulary", ha="center", fontsize=8, color=GREY)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "architecture_diagram.png"), dpi=160)


if __name__ == "__main__":
    os.makedirs(FIGURES, exist_ok=True)
    torch.set_num_threads(4)
    diagram()
    for make in (loss_curves, architecture, context, temperature, attention):
        try:
            make()
        except FileNotFoundError as e:
            print(f"skipped {make.__name__}: {e.filename} is missing")
    print(f"figures written to {FIGURES}")
