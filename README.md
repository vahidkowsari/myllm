# myllm — a tiny LLM you can actually read

A small, heavily-commented character-level GPT in Apple's [MLX](https://github.com/ml-explore/mlx).
The goal is **understanding**: every layer that makes a modern LLM work is here, written plainly,
small enough to train on an Apple-Silicon laptop (runs on the Mac GPU via MLX) in a few minutes.

It is deliberately in the spirit of Karpathy's nanoGPT, but the code is annotated so you can
trace a single character all the way from text → token → embedding → attention → logits → loss
→ gradient, and back.

> **Full walkthrough:** see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for an end-to-end
> reference — the math, every module, the training/sampling internals, and a complete config
> reference. A slide deck is in [`docs/presentation.html`](docs/presentation.html).

## The layers, bottom to top

This is the whole stack of "what is happening at the lowest layers of an LLM":

1. **Tokenizer** (`data.py`) — turns text into integers. We use a *character-level* tokenizer
   (one token per character) so there's zero magic: the vocab is just the set of unique
   characters in the training file.
2. **Token embedding** (`model.py`) — each token id looks up a learned vector. Position is
   *not* a separate table; it's injected inside attention by RoPE (below).
3. **Self-attention + RoPE** (`model.py: CausalSelfAttention`) — every position builds a Query,
   Key, and Value; attention is `softmax(Q·Kᵀ / √d) · V`, masked so a position can only see
   the past. **RoPE** rotates each token's Q/K by its position, so the dot product encodes how
   far apart two tokens are. This is where tokens "talk to each other."
4. **SwiGLU feed-forward** (`model.py: MLP`) — a per-token gated network (`silu(gate(x))·up(x)`
   → `down`) that does the "thinking" after tokens have mixed information.
5. **Transformer block** (`model.py: Block`) — attention + SwiGLU, each wrapped in an
   **RMSNorm** and a residual (skip) connection. Stack N of these.
6. **LM head** (`model.py`) — a final linear layer (weight-tied to the token embedding)
   projecting back to vocab size, giving a logit (score) for every possible next character.
7. **Loss** — cross-entropy between predicted next-char distribution and the actual next char.
8. **Training loop** (`train.py`) — sample random chunks of text, predict the next char at
   every position, backprop, step the optimizer.
9. **Sampling** (`sample.py`) — feed a prompt, get logits for the next char, sample from the
   softmax, append, repeat. A **KV cache** in `generate()` makes each new char one forward
   step over a single token instead of re-reading the whole context.

This is a ~2024-era transformer, not the original 2019 GPT-2: it uses RoPE (not learned
position embeddings), RMSNorm (not LayerNorm), a SwiGLU feed-forward (not a GELU MLP), and a
KV cache at inference. See the top of `model.py` for why each is the modern default.

## Quick start

```bash
# 1. set up the environment (creates ./.venv and installs mlx)
./setup.sh

# 2. activate it
source .venv/bin/activate

# 3. download + tokenize Tiny Shakespeare (~1MB)
python data.py

# 4. train (writes ckpt.npz + ckpt.json). ~7 minutes on a Mac.
python train.py

# 5a. generate text from the trained model (one-shot)
python sample.py --prompt "ROMEO:"

# 5b. ...or run it interactively: loads once, then loop typing prompts and watch it stream
python sample.py -i

# 5c. ...or serve it over HTTP so other programs can call it (stdlib only, no extra deps)
python serve.py            # then: curl -s localhost:8000/complete -d '{"prompt": "Once"}'
```

> It's a *character-level Shakespeare continuer*, not a chatbot — give it the start of a line
> and it keeps writing in that style.

### Train on your own data

Point it at a folder of text files and it builds the corpus for you — no code changes:

```python
# in config.py
config.data_dir  = "./mydata"   # every *.txt under here becomes the training corpus
config.tokenizer = "bpe"        # subwords beat char-level on real prose/code
config = medium()               # (optional) the larger ~20-25M preset
```

```bash
python train.py            # trains on your corpus
python serve.py            # serve the result
```

A 2.7–25M model won't be an assistant, but on focused data it makes a surprisingly decent
domain-specific autocomplete / text generator.

## Bonus: a multimodal (image → text) demo

The same GPT can *see*. A transformer doesn't really process text — it processes a sequence of
vectors, and the only thing that turns text into those vectors is one embedding line. So to go
multimodal you just build a second converter that turns an image into the same kind of vectors
and splice them into the sequence; attention does the rest.

```bash
python vision.py        # peek at the toy dataset: colored shapes drawn + captioned in your terminal
python train_mm.py      # train an image->caption model from scratch (~2 min); captions fresh images
python sample_mm.py --n 8   # caption 8 brand-new random images
```

`vision.py` chops each image into patches, projects each patch to one model vector (`PatchEmbed`,
the image analogue of the token embedding), and prepends them to the text via GPT's `prefix=`
hook. Loss is graded only on the caption characters — the same masking trick as SFT. See the long
comment at the top of `vision.py`.

## Files

| file        | what it is |
|-------------|------------|
| `data.py`   | downloads the corpus, builds the tokenizer, makes train/val tensors, saves/loads vocab |
| `bpe.py`    | a from-scratch byte-level BPE tokenizer (used when `tokenizer = "bpe"`) |
| `model.py`  | the GPT itself — embeddings, attention, feed-forward, blocks, head. **Read this first.** |
| `train.py`  | the pretraining loop (`mx.compile`, warmup→cosine LR, weight decay) + loss reporting |
| `sft.py`    | post-training: instruction-tune the base model on loss-masked (instruction, response) pairs |
| `serve.py`  | a tiny stdlib HTTP completion API (`POST /complete`, with optional SSE streaming) around a checkpoint |
| `sample.py` | autoregressive generation (`-i` = interactive REPL, `--chat` = instruction mode) |
| `vision.py` | multimodal demo: synthetic shape images, `PatchEmbed` (image→vectors), and the captioning model |
| `train_mm.py`| trains the image→caption model from scratch (loss-masked captions, like SFT) |
| `sample_mm.py`| captions fresh random images, drawn in color in the terminal |
| `config.py` | one place for all hyperparameters (model size, lr, tokenizer, MoE/GQA toggles, etc.) |

## Knobs to play with

Everything is in `config.py`.

- **Size.** Raise `n_layer`, `n_head`, `n_embd`, `block_size` to make it "smarter" — and watch
  training slow down. That trade-off *is* the lesson of LLM scaling, in miniature. Defaults are
  ~2.7M params so it trains fast.
- **Tokenizer.** `tokenizer = "bpe"` (with `bpe_vocab_size`) swaps the char tokenizer for a
  subword one, so the same `block_size` covers far more text.
- **Data.** Point `data_url` / `data_path` at a different corpus to train on something other
  than Shakespeare.
- **Grouped-query attention.** Set `n_kv_head` below `n_head` (must divide it) to shrink the
  KV cache.
- **Mixture of experts.** Set `use_moe = True` (with `n_experts`, `n_experts_per_tok`) to make
  each block's feed-forward a sparse MoE.
- **Attention tweaks (recent papers).** `use_qk_norm` (RMSNorm Q/K before scoring — training
  stability) and `use_softmax1` (softmax-off-by-one, so a token can attend to "nothing").
- **Scale it up.** `dtype = "bfloat16"` (half-precision weights) and `use_grad_checkpoint = True`
  (recompute activations in backward) let a much bigger model fit on the laptop. `config.medium()`
  is a ready ~20-25M-param preset that turns both on — swap the last line of `config.py` to
  `config = medium()`. The tokenized corpus is cached to disk and memory-mapped, so it scales to
  corpora bigger than RAM.

Sampling controls (flags on `sample.py`): `--temperature`, `--top_k`, `--top_p` (nucleus),
`--repetition_penalty`, `--tokens`, `--entropy` (adapt sampling to the model's own uncertainty),
`--quantize 4`/`8` (n-bit weights — ~6× smaller, still coherent).
