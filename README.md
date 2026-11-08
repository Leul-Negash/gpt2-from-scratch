# GPT-2 from scratch

I implemented a small GPT-2-style language model from scratch in PyTorch and trained it on Tiny Shakespeare:
my own byte-level BPE tokenizer, causal multi-head self-attention, the feed-forward network, LayerNorm, residual
connections, Transformer blocks, a full training pipeline with checkpoints, and greedy and temperature-based text
generation. I then used it to study temperature, prompt length and context, and the effect of the number of
attention heads, the number of layers, the type of positional encoding and the causal mask.

The technical report, which explains *Attention Is All You Need* and connects every concept to this code, is in
**[REPORT.md](REPORT.md)**.

## Results

The main model (875,264 parameters) trained for 5,000 steps (about 70 minutes on 4 CPU threads) and reached a
**validation loss of 2.784** (perplexity 16.2, from 6.25 = ln 512 at the start). A sample at temperature 0.8:

```
ROMEO:
As the rovil, with a pawdy palace:
As that few, did I see the law:
Whilst thou remember thy heart's purposed times;
And mayst thou draw thy soul bed,
Show her from the breaking gentleman:
So, Richard, our father's life,
I am favour'd nor no more in the matter of mine.
```

Model and training configuration (`config.py`; also saved in `results/gpt_history.json`):

| Setting | Value | Setting | Value |
|---|---|---|---|
| layers | 4 | optimiser | AdamW, β = (0.9, 0.95), weight decay 0.1 |
| attention heads | 4 (32 dimensions each) | learning rate | 1e-3, 200 warm-up steps, cosine decay to 1e-4 |
| embedding width | 128 | batch | 32 sequences × 128 tokens |
| context length | 128 tokens | steps | 5,000 |
| vocabulary | 512 (byte-level BPE) | dropout | 0.1 |
| positions | learned | gradient clipping | 1.0 |
| parameters | 875,264 | data | Tiny Shakespeare, 516,405 train / 59,401 validation tokens |

The full loss history is in `results/train_gpt.log`, and more samples are in `results/samples.txt`.

| Experiment | Main finding |
|---|---|
| Train vs validation loss | both fall the whole run; the gap grows from 0.27 (step 1000) to 0.48, so the model starts to memorise |
| Temperature | greedy (T = 0) loops (10.5% repeated 4-word sequences); T = 0.5–0.8 gives the best text; at T = 2 only 60% of the words are real |
| Context length | loss falls from 3.76 with 1 token of context to 2.72 with 32, then stays flat: the model uses about the last 32 tokens |
| Prompt length | a longer prompt sets the speaker and the style for the first lines, then the model falls back to its own habits |
| Layers (1 / 2 / 4 / 6) | validation 3.509 / 3.466 / 3.437 / 3.412 after 800 steps: depth helps, with diminishing returns |
| Heads (1 / 2 / 4 / 8) | all within 0.015 at width 128: no measurable effect at this size |
| Positions | none 3.522, learned 3.437, unscaled sinusoidal 4.382 (the fixed sinusoids drown out the small token embeddings) |
| No causal mask | validation 3.197 looks better, but at the last position, where there is no future to copy, 3.846 vs 3.470 |

| Figure | |
|---|---|
| [loss curves](figures/loss_curves.png) | [temperature](figures/temperature.png) |
| [context length](figures/context.png) | [architecture experiments](figures/architecture.png) |
| [attention heads](figures/attention.png) | [architecture diagram](figures/architecture_diagram.png) |

## What is implemented where

| File | What it does |
|---|---|
| `tokenizer.py` | byte-level BPE tokenizer: training (merging frequent pairs), encoding, decoding, saving |
| `model.py` | `LayerNorm`, `gelu`, sinusoidal positions, `CausalSelfAttention` (multi-head), `FeedForward`, `Block`, `GPT` with `generate` |
| `config.py` | model and training settings, each with the reason for its value |
| `data.py` | downloads Tiny Shakespeare, trains the tokenizer, 90/10 split, input-target batches |
| `train.py` | training loop with AdamW, warm-up + cosine schedule, validation, checkpoints, resume |
| `generate.py` | text generation from a checkpoint (greedy or temperature sampling, optional top-k) |
| `experiments.py` | temperature, context and prompt length, architecture experiments |
| `make_figures.py` | all figures in `figures/` |
| `tests/test_model.py` | 16 tests: tokenizer, LayerNorm, GELU, attention against PyTorch, causality, generation, resume |

From PyTorch I only use tensors, autograd, `nn.Linear`, `nn.Embedding`, `nn.Dropout`, softmax, cross-entropy and
AdamW. No pretrained models, no Hugging Face, no `nn.Transformer`, `nn.MultiheadAttention` or `nn.LayerNorm`.
PyTorch's built-in attention and LayerNorm appear only in the tests, to check that mine give the same numbers.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dataset (1.1 MB) is downloaded automatically on the first run. Everything runs on a CPU.

## Training

```bash
python train.py                        # the main model: 5,000 steps, about 70 minutes on 4 CPU threads
python train.py --steps 1000 --name quick
python train.py --stop_at 2500         # pause at step 2500 ...
python train.py --resume               # ... and continue exactly where it stopped
python train.py --n_head 8 --n_layer 6 --position sinusoidal --name my_variant
```

Every 250 steps it prints the training and validation loss. `checkpoints/<name>_best.pt` keeps the model with the
lowest validation loss; `checkpoints/<name>_last.pt` keeps everything needed to resume (weights, optimiser state,
step, loss history, random states). The loss history is also saved to `results/<name>_history.json`.

## Generating text

The trained model is included as `checkpoints/gpt_best.pt`, so this works without training:

```bash
python generate.py                                   # prompt "ROMEO:", temperature 0.8
python generate.py --prompt "KING HENRY:" --greedy
python generate.py --prompt "To be, or not" --temperature 1.2 --tokens 300 --seed 3
python generate.py --temperature 0.8 --top_k 20
```

## Experiments, figures and tests

```bash
python experiments.py temperature            # uses the trained model, about a minute
python experiments.py context
python experiments.py architecture           # 10 models of 800 steps each, about 1.5 hours; prints saved results if they exist
python make_figures.py
python -m pytest -v
```

## Design decisions

- **Decoder-only, pre-LayerNorm, learned positions, GELU, weight tying, residual projections initialised with
  std 0.02/√(2·n_layer)**: the GPT-2 design, so the model is GPT-2 in structure, only smaller.
- **Small configuration** (4 layers, 4 heads, width 128, context 128, 875k parameters): it trains in about 70 minutes
  on a laptop CPU, and with 516k training tokens a much bigger model would only memorise the text faster. All sizes
  are powers of two.
- **Byte-level BPE with 512 tokens**: the GPT-2 tokenizer type, trained on my own data. 512 rather than GPT-2's 50,257
  because the dataset is 1 MB: every token is seen often enough to learn its embedding. The tokenizer is trained on
  the training split only.
- **AdamW (β = 0.9, 0.95), weight decay 0.1 on matrices, clipping at 1.0, warm-up then cosine decay**: standard GPT
  training practice. Peak learning rate 1e-3 because small models tolerate larger steps than GPT-2's 2.5e-4.
- **Dropout 0.1**: as in GPT-2 and the original Transformer; the dataset is small enough to overfit without it.
- **Contiguous 90/10 split**: the validation text is a later part of the plays, so no line appears on both sides.
- **Fixed evaluation batches**: train and validation loss are measured on the same windows every time, so changes in
  the curve come from learning, not from which batches were drawn.

## References

- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS 30*. https://arxiv.org/abs/1706.03762
- Radford, A., et al. (2019). Language models are unsupervised multitask learners (GPT-2). OpenAI.
- Karpathy, A. (2022). Let's build GPT: from scratch, in code, spelled out. https://www.youtube.com/watch?v=kCc8FmEb1nY
- Alammar, J. (2018). The illustrated Transformer. https://jalammar.github.io/illustrated-transformer/
- Sennrich, R., Haddow, B., Birch, A. (2016). Neural machine translation of rare words with subword units (BPE). *ACL*.
- Ba, J. L., Kiros, J. R., Hinton, G. E. (2016). Layer normalization. arXiv:1607.06450.
- He, K., et al. (2016). Deep residual learning for image recognition. *CVPR*.
- Hendrycks, D., Gimpel, K. (2016). Gaussian error linear units (GELUs). arXiv:1606.08415.
- Karpathy, A. Tiny Shakespeare dataset, from char-rnn. https://github.com/karpathy/char-rnn
- PyTorch documentation. https://docs.pytorch.org/docs/stable/index.html
