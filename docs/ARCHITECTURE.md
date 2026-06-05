# myllm — Architecture & Reference

A complete walkthrough of how `myllm` works, end to end. The source files are heavily
commented and meant to be read directly; this document is the map that ties them together and
the reference for every knob. If you only read one source file, read `model.py`.

`myllm` is a small, educational GPT written in Apple's [MLX](https://github.com/ml-explore/mlx).
It is a ~2024-era transformer (RoPE, RMSNorm, SwiGLU, KV cache, optional GQA and MoE) kept
deliberately tiny (~2.7M params by default) so it trains in minutes on an Apple-Silicon laptop.
Every design choice optimizes for *understanding* over performance.

## Contents

1. [The big picture](#1-the-big-picture)
2. [Data & tokenization](#2-data--tokenization)
3. [The model](#3-the-model-modelpy)
4. [Training (pretraining)](#4-training-trainpy)
5. [Sampling / generation](#5-sampling--generation-samplepy)
6. [Post-training (SFT)](#6-post-training-sft-sftpy)
7. [Configuration reference](#7-configuration-reference-configpy)
8. [Checkpoint format](#8-checkpoint-format)
9. [Parameter budget](#9-parameter-budget)
10. [What's modern vs. GPT-2](#10-whats-modern-vs-gpt-2)
11. [Glossary](#11-glossary)

---

## 1. The big picture

An LLM is a function that, given a sequence of tokens, predicts the next one. Training nudges
that function until its predictions match real text; sampling runs it forward repeatedly to
generate.

```
text ──► tokenizer ──► token ids (integers)
                          │
                          ▼
              ┌───────────────────────────┐
              │ token embedding (B,T,C)    │  "what is each token?"
              └───────────────────────────┘
                          │   (position is injected later, inside attention, via RoPE)
                          ▼
              ┌───────────────────────────┐
              │  N × Transformer Block     │
              │   x += attn(rmsnorm(x))    │  tokens mix information
              │   x += ffn(rmsnorm(x))     │  each token "thinks"
              └───────────────────────────┘
                          │
                          ▼
                   final RMSNorm
                          │
                          ▼
              LM head (tied to token embedding) ──► logits (B,T,vocab)
                          │
            ┌─────────────┴──────────────┐
       training: cross-entropy      inference: sample next token, append, repeat
       vs. the true next token
```

Shapes are written throughout the code as `(B, T, C)`:

| symbol | meaning |
|--------|---------|
| `B` | batch size (sequences processed in parallel) |
| `T` | time / sequence length (number of tokens, ≤ `block_size`) |
| `C` | channels = `n_embd`, the width of the model |

MLX specifics that shape the code:

- **Unified memory, no devices.** MLX runs on the Apple-Silicon GPU automatically — there is no
  `.to(device)`. Arrays just live in one place.
- **Lazy evaluation.** Operations build a computation graph; nothing actually computes until you
  call `mx.eval(...)` (or pull a Python value out with `.item()` / `print`). The training and
  sampling loops call `mx.eval` at the points where work should happen.
- **A module is a dict.** Any `mx.array` attribute on an `nn.Module` is a trainable parameter.

---

## 2. Data & tokenization

> Files: `data.py`, `bpe.py`

A model never sees text — it sees integers. Tokenization is the bottom layer: text ↔ integers.

### Char-level tokenizer (default)

`CharTokenizer` (`data.py`) is maximally transparent: **one token = one character**. The
vocabulary is literally `sorted(set(text))`, so for Tiny Shakespeare it's ~65 unique characters.
`encode` maps a string to a list of ids; `decode` maps ids back. Unknown characters at sampling
time are silently dropped so a stray keystroke can't crash the prompt loop.

Pros: zero magic, tiny vocab. Con: the model spends its whole context window on individual
letters — `block_size = 128` only sees 128 characters.

### BPE tokenizer (optional)

Set `tokenizer = "bpe"` in `config.py` to use `bpe.py`, a from-scratch **byte-level
Byte-Pair Encoding** tokenizer (in the spirit of Karpathy's minbpe). The algorithm:

1. Start from the raw bytes of the text (256 base byte tokens).
2. Find the most frequent adjacent pair of tokens and **merge** it into one new token.
3. Repeat until the vocab reaches `bpe_vocab_size`.

So common chunks like `the`, `" and"`, `ing` become single tokens. One token now covers several
characters, and the same `block_size` stretches over far more text (~2× on Shakespeare at vocab
512; more with a larger vocab). `decode` glues each token's bytes back together and UTF-8 decodes.

Key functions in `bpe.py`: `get_stats` (count adjacent pairs), `merge` (replace a pair),
`BPETokenizer.train` / `.encode` / `.decode`. Run `python bpe.py` for a standalone demo.

### Batching

`get_batch(data, block_size, batch_size)` (`data.py`) picks `batch_size` random start positions
in the corpus and gathers a `(batch_size, block_size)` grid of windows. The target `y` is simply
the input `x` shifted left by one: at every position `t`, the model must predict token `t+1`
from tokens `0..t`. That single shift is the entire supervised signal.

### Tokenized-corpus cache (memory-mapped)

`load_tokens` tokenizes the whole corpus **once** and writes the ids to a binary blob
(`<corpus>.<tokenizer>.tokens.bin`, uint16) plus a JSON sidecar, then returns a **`np.memmap`** of
it. Memory-mapping means the ids stay on disk and the OS pages in only the windows `get_batch`
actually samples — so a corpus larger than RAM still trains, and re-runs skip re-encoding. The
cache is keyed by tokenizer + vocab size + corpus length and rebuilt automatically if any change.
`get_batch` selects start positions with MLX's RNG (so `config.seed` still controls batching),
gathers those windows from the memmap, and moves just that small batch onto the GPU as int32.

### Serialization

Both tokenizers expose `to_meta()` (serialize) and `from_meta()` (rebuild), and
`data.load_tokenizer(meta)` dispatches on the saved `type` field. This is how `sample.py`
reconstructs the exact tokenizer from `ckpt.json` without re-reading the corpus.

---

## 3. The model (`model.py`)

The core file, written bottom to top. Below, each component in the order it appears.

### 3.1 Token embedding

`nn.Embedding(vocab_size, n_embd)` — a lookup table turning each token id into a learned
`C`-dimensional vector. There is **no position embedding table**; position is supplied by RoPE
inside attention (below).

### 3.2 RoPE — rotary position embeddings

Attention is order-blind: shuffle the tokens and the math is unchanged. RoPE injects position by
**rotating** each token's Query and Key vectors by an angle proportional to its position. Because
a rotation by angle `θ_i` followed by a dot product depends only on the *difference* of the two
positions, the model naturally perceives *relative distance* — and it generalizes to distances
it never saw at exactly that absolute position.

- `rope_tables(seq_len, head_dim, base)` precomputes `cos`/`sin` of the per-dimension-pair angles.
  Frequencies are geometrically spaced (`base^(-i/half)`): fast-spinning pairs first, slow last —
  like the hands of many clocks. `base` is `rope_base` (default 10000).
- `apply_rope(x, cos, sin)` applies the 2-D rotation to each dimension pair:
  `x1' = x1·cos − x2·sin`, `x2' = x2·cos + x1·sin`.

`head_dim` must be even (asserted) because RoPE rotates dimensions in pairs.

### 3.3 Causal self-attention (`CausalSelfAttention`)

The one mechanism that makes a transformer a transformer. Each token produces a **Query** ("what
am I looking for?"), a **Key** ("what do I offer?"), and a **Value** ("what do I pass on?"). A
token's new representation is a weighted average of the Values of all earlier tokens, weighted by
how well its Query matches each Key:

```
attention = softmax( Q · Kᵀ / √head_dim )      # (T, T), one row per token
output    = attention · V
```

- **Causal mask.** Each query at absolute position `p` may attend only to keys at positions
  `≤ p`. Future positions get `-inf` before the softmax, so they contribute ~0. The mask is built
  generally to also handle the KV cache (see §5): `mask[i, j] = (j ≤ offset + i)`.
- **Multi-head.** The `C` channels are split into `n_head` independent sub-spaces; attention runs
  in each, so heads can specialize. They are merged back at the end via the output projection.
- **Written longhand on purpose.** The explicit `Q·Kᵀ/√d`, mask, and softmax are kept visible —
  MLX ships a fused `mx.fast.scaled_dot_product_attention`, but seeing the steps is the point. Do
  not replace it.

#### Grouped-query attention (GQA)

By default `n_kv_head == n_head` → ordinary multi-head attention. Set `n_kv_head` **lower** (it
must divide `n_head`) and several query heads share each K/V head. Q gets `n_head` heads; K and V
get `n_kv_head` heads. After RoPE and the cache concat, `repeat_kv` expands the K/V heads to line
up with the query heads for the score computation. The win: the **KV cache stores the smaller
K/V tensors**, which is the dominant inference-memory cost on long contexts. Used by Llama-2-70B,
Mistral, Qwen, etc.

Because Q and K/V can have different head counts, the projections are **separate** (`q_proj`,
`kv_proj`) rather than one fused `qkv` matrix.

#### Two optional attention tweaks (recent papers)

Both are off by default and config-gated; each travels with the checkpoint (the flag lives in the
saved `config`), so sampling rebuilds the exact same architecture.

- **QK-Norm** (`use_qk_norm`). Apply an `RMSNorm` to each head's Query and Key vectors (over
  `head_dim`) **before** RoPE/scoring. This bounds the magnitude of `Q·Kᵀ`, so the attention
  logits can't blow up — a cheap stability win that lets you push the learning rate higher without
  divergence. Q and K get their own learnable gains (two `RMSNorm` modules), adding a handful of
  params. Now standard in many 2024-25 models.
- **Softmax-off-by-one** (`use_softmax1`, a.k.a. *softmax1* / *quiet attention*). Add a phantom
  `+1` to the softmax denominator — `softmax1(x)_i = exp(x_i) / (1 + Σ exp(x_j))` — as if there
  were one extra logit pinned at 0. A normal softmax forces every attention row to sum to exactly
  1, so a token must spend all its attention somewhere even when nothing is relevant; that pressure
  surfaces as a few giant outlier activations (bad for quantization). The `+1` lets a row sum to
  **less than 1**, i.e. attend to "nothing" — which is also how *attention sinks* form. Implemented
  as `softmax_off_by_one` (numerically stable: the phantom logit 0 is folded into the row-max).
  Zero extra params.

### 3.4 RMSNorm

`nn.RMSNorm(n_embd)` rescales each token vector by its root-mean-square, then multiplies by a
learned per-channel gain. Like LayerNorm but with no mean-subtraction and no bias — cheaper, and
just as effective in practice. Applied **pre-norm**: before each sub-layer, inside the residual.

### 3.5 Feed-forward — SwiGLU (`MLP`)

After attention mixes information across tokens, the feed-forward network is where each token
"thinks" on its own. A plain MLP is `fc → GELU → proj`. **SwiGLU** adds a *gate*: a second linear
runs in parallel and multiplies the activated branch, letting the network suppress or pass
through each hidden unit per token:

```
hidden = silu(gate(x)) · up(x)        # silu(z) = z · sigmoid(z)
out    = down(hidden)
```

It uses three matrices instead of two, so the hidden width is shrunk to `~8/3·C` (Llama's trick)
to keep the parameter count close to a `4·C` plain MLP (`n_embd=192 → hidden=512`).

### 3.6 Mixture of experts — MoE (optional)

Set `use_moe = True` and each block's feed-forward becomes a **sparse MoE**: `n_experts` separate
SwiGLU networks plus a tiny **router** (one Linear). For each token the router scores every
expert; the top `n_experts_per_tok` (`k`) are kept, their scores softmaxed into mixing weights,
and the token's output is the weighted sum of just those experts. The model holds many experts'
worth of *parameters* but each token only uses a few — capacity without proportional compute
(the idea behind Mixtral).

Two things make it work, both in `MoE.__call__`:

- **Top-k routing.** `argsort` the router logits, take the top `k`, re-softmax.
- **Load-balancing auxiliary loss.** Without it the router collapses onto one favorite expert and
  the rest never learn. The Switch-Transformer aux loss `E · Σ(f_e · P_e)` — where `f_e` is the
  fraction of tokens choosing expert `e` and `P_e` is the mean router probability for it — is
  minimized when load is spread evenly. It is returned from the block and added to the training
  loss, scaled by `moe_aux_coef`.

> **Pedagogical simplification:** for clarity the implementation runs *every* token through
> *every* expert and then zeros out the ones a token didn't pick. That discards MoE's compute
> savings (a real implementation gathers only the routed tokens per expert) but keeps the routing
> math readable, and is harmless on a tiny model. This is called out in the code.

### 3.7 Transformer block (`Block`)

```
x = x + attention(rmsnorm(x))
x = x + feedforward(rmsnorm(x))
```

The residual `x +` lets gradients flow straight through, so many blocks stack without the signal
vanishing. RMSNorm keeps each token vector at a sane scale before each sub-layer. The
feed-forward is a dense `MLP` by default, or `MoE` if `use_moe`. The block also threads the KV
cache through attention and returns the MoE aux loss (0 for the dense path).

### 3.8 The full model (`GPT`)

Embedding → `n_layer` blocks → final RMSNorm → LM head.

- **Weight tying.** The output projection (the "LM head") reuses the token embedding matrix via
  `token_emb.as_linear(x)` — the same matrix maps ids→vectors on the way in and vectors→logits on
  the way out. This saves parameters and tends to help. There is deliberately **no separate
  `lm_head` weight** to keep in sync.
- **Loss.** Cross-entropy at every position between the predicted next-token distribution and the
  true next token, averaged. Plus the MoE aux loss (0 unless `use_moe`).
- **Initialization (`_init_weights`).** GPT-2 recipe: normal, std 0.02. The output projection of
  each residual sub-layer (attention's `proj` and the feed-forward's `down`) is additionally
  scaled by `1/√(2·n_layer)` so the residual stream's variance doesn't grow with depth —
  stabilizing training. RMSNorm gains stay at their default 1.0.

`GPT.__call__(idx, targets=None, caches=None)` returns `(logits, loss, new_caches)`.

---

## 4. Training (`train.py`)

The job of training: repeatedly (1) grab a random batch, (2) predict the next token at every
position, (3) measure loss, (4) backprop, (5) step the optimizer.

### One-call forward + backward

MLX has no `loss.backward()`. `nn.value_and_grad(model, loss_fn)` returns a function that runs
the forward pass **and** computes gradients of the loss w.r.t. every parameter in one shot.

### The compiled step

The entire step — forward, backward, grad clip, optimizer update, weight decay — is wrapped in
`mx.compile`, which traces the lazy graph once and fuses it into a single optimized kernel (a
significant speedup). Model and optimizer state are captured as both `inputs` and `outputs` so
MLX knows those arrays are updated in place across calls:

```python
state = [model.state, optimizer.state]

@partial(mx.compile, inputs=state, outputs=state)
def step(x, y):
    loss, grads = loss_and_grad(model, x, y)
    grads, _ = optim.clip_grad_norm(grads, config.grad_clip)   # cap gradient spikes
    optimizer.update(model, grads)
    # decoupled weight decay on weight matrices only (see below)
    ...
    return loss
```

### Learning-rate schedule

A **warmup → cosine decay** schedule (standard for transformers): linearly ramp the LR from 0
over the first `warmup_iters` steps (so early, half-random gradients don't blow things up), then
cosine-decay down to `min_lr` over the rest. Built with `optim.linear_schedule`,
`optim.cosine_decay`, and stitched at the warmup boundary with `optim.join_schedules`. The live
status line prints the current LR so you can watch it ramp and ease off.

### Selective (decoupled) weight decay

Weight decay should pull the big weight **matrices** toward zero but **not** the 1-D RMSNorm
gains, biases, or the token embedding — shrinking those just hurts. MLX's optimizers apply decay
uniformly, so AdamW's built-in `weight_decay` is turned **off** and decay is applied manually:

- `build_decay_mask` precomputes a flat `{param_name: 1.0 or 0.0}` map. The rule "2-D **and** not
  the token embedding" cleanly selects exactly the Linear weights.
- Inside the step: `w ← w − lr · weight_decay · mask · w`.

This is the "W" in AdamW, made selective.

### Scaling up: precision & memory

Two config knobs let a much bigger model train on the same laptop (`config.medium()` flips both on
for a ~20-25M-param preset):

- **`dtype="bfloat16"`.** Right after init the model is cast to bf16 (`tree_map(astype)`), so
  weights *and* optimizer moments live in half precision — ~2× less memory and faster matmuls.
  The one place low precision bites, the softmax/log inside the loss, is kept in float32 by
  upcasting the logits in `GPT.__call__`. (Teaching simplification: real mixed precision also
  keeps an fp32 *master* copy of the weights for the update; here the bf16 weights are the master.)
- **`use_grad_checkpoint=True`.** Each block is wrapped in `nn.utils.checkpoint` during training,
  which **discards the block's internal activations and recomputes them in the backward pass**.
  That trades ~30% more compute for a large drop in peak memory — the lever that lets you go deeper
  without running out of room. `nn.utils.checkpoint` (not bare `mx.checkpoint`) is used so
  gradients still reach the block's *parameters*; it is a no-op at inference (no backward).

### Monitoring & checkpoint

- `estimate_loss` averages the loss over `eval_iters` batches on both train and val splits every
  `eval_interval` steps (a single batch's loss is too noisy; watching val loss reveals
  overfitting). Dropout is turned off during measurement.
- On finish, weights are saved with `mx.savez` to `ckpt.npz`, and the tokenizer + config to
  `ckpt.json` (see §7).

---

## 5. Sampling / generation (`sample.py`)

Generation is autoregressive: forward pass → logits for the next token → pick one → append →
repeat. `GPT.generate` drives the loop; `sample.py` wraps it with a CLI and an interactive REPL.

### The KV cache

Naively, every step re-reads the entire context (O(T²) over a sequence). Instead each attention
layer **remembers its past Keys and Values**, so a new token is one forward pass over a *single*
token (O(T) per token). Implementation:

- `CausalSelfAttention` accepts a `(past_k, past_v)` cache, concatenates the new K/V, and returns
  the updated cache. RoPE uses the cache length as a position `offset` so new tokens get the right
  absolute positions. Under GQA the cache holds the smaller `n_kv_head` tensors.
- `generate` keeps a per-layer cache and feeds **only the newest token** on the fast path. It
  rebuilds the cache from scratch only when there isn't one yet, or when the sequence grows past
  `block_size` — re-priming on the last `block_size` tokens keeps every RoPE position inside
  `[0, block_size)`, the range the model trained on.

> **Limitation:** because re-priming keeps positions in range, generating *past* `block_size`
> falls back to O(T²) per step. Fine for a tiny model; documented in `generate`'s docstring.

### Sampling controls

Applied per step, in this order:

| control | flag | effect |
|---------|------|--------|
| repetition penalty | `--repetition_penalty` (1.0 = off) | dampens logits of tokens already in the context (HF convention: divide positive / multiply negative logits by the penalty) to avoid loops |
| temperature | `--temperature` (0.8) | divides logits; <1 sharpens (more confident), >1 flattens (more random) |
| top-k | `--top_k` (40, 0 = off) | keep only the `k` most likely tokens |
| top-p (nucleus) | `--top_p` (0 = off) | keep the smallest set of tokens whose probabilities sum to `p`; adapts the candidate count per step |
| quantize | `--quantize` (0 = off; 2/3/4/6/8) | n-bit weights for smaller/faster inference (see below); orthogonal to the sampling controls |

Then a token is **sampled** (not argmaxed) from the surviving logits via
`mx.random.categorical`, so output has variety.

### Quantized inference (`--quantize`)

`sample.py --quantize {2,3,4,6,8}` shrinks the weights *after* loading the float checkpoint, with
MLX's `nn.quantize`. Each weight matrix is split into groups of `--q_group_size` (64) columns;
each group is stored as small ints plus a scale, so a 4-bit model is ~1/6–1/8 the size of float32
and decodes faster. A `class_predicate` quantizes only `Linear`/`Embedding` layers whose last
dimension divides the group size (this skips tiny oddballs like the MoE router), and the **tied LM
head comes along for free** — it reuses the now-`QuantizedEmbedding` weight via `as_linear`. To
keep matrices quantization-friendly at any size, the SwiGLU hidden width is rounded up to a
multiple of 64 (a no-op at the default `n_embd=192`, where it is already 512). On the default model
4-bit is ~6.2× smaller (10.7 MB → 1.7 MB) and still writes coherent stories — and `use_softmax1`
helps here, since the outlier activations it suppresses are exactly what make quantization lossy.

### Entropy-based ("entropix") sampling

`--entropy` switches to a self-paced sampler (`GPT._entropy_sample`) that adapts to the model's
*own* uncertainty instead of a fixed temperature — when on, it **bypasses** `--temperature`,
`--top_k`, and `--top_p` (the repetition penalty still applies first). From the next-token
distribution `p` it measures two quantities, both in nats:

- **entropy** `H = -Σ p·log p` — how spread out the distribution is overall.
- **varentropy** `Σ p·(-log p − H)²` — the *variance of the surprisal*: is the model torn between a
  few sharp options (low) or genuinely vague (high)?

The rule: if **both** are below their thresholds the model is confident → take the **argmax**
(greedy); otherwise sample at a temperature that **rises** with entropy and varentropy, so a more
uncertain model explores more. Knobs live in `config.py` (`ent_low`, `vent_low`, `ent_base_temp`,
`ent_ent_coef`, `ent_vent_coef`). This makes the model's uncertainty visible in how the text is
generated — confident stretches go deterministic, vague ones loosen up.

### Streaming

`generate` accepts an `on_token` callback invoked with each token id as it's produced.
`sample.py -i` uses it to print characters live while the KV cache is reused internally — one
forward step per character instead of re-reading the prompt each time. (With BPE, a token that is
a partial multibyte UTF-8 sequence renders as `�` mid-stream; harmless for ASCII corpora.)

---

## 6. Post-training (SFT) (`sft.py`)

Everything above produces a **base model** — a text continuer. **Post-training** is what turns
it into something that follows instructions. The first (and most important) stage is
**Supervised Fine-Tuning (SFT)**, a.k.a. instruction tuning. The full modern recipe is
*pretrain → SFT → preference tuning (DPO/RLHF)*; `myllm` implements the first two stages of
that arc conceptually, with SFT in code.

SFT is just pretraining with **two changes**:

1. **Data shape.** Instead of raw text, use `(instruction, response)` pairs wrapped in a fixed
   template the model learns to recognize:
   ```
   Instruction:
   {instruction}

   Response:
   {response}<|endoftext|>
   ```
   The template uses only characters in the corpus vocab (no `#`), and ends the response with
   the same `<|endoftext|>` marker the base model already saw between stories — so it doubles as
   a learned "stop generating" signal.

2. **Loss masking.** Grade the model **only on the response tokens**. It should learn to
   *produce* the response, not to echo the instruction back. So the cross-entropy for every
   instruction/header token is multiplied by 0; only response tokens (plus the end marker)
   count. This single mask is what makes SFT "supervised" in the instruction sense.

Because there's no human instruction dataset here, `sft.py` **synthesizes** one from the same
TinyStories corpus: split it on `<|endoftext|>` into individual stories, pick each story's most
frequent content word as its topic, and form the pair `("Write a story about {topic}.", story)`.
The pairs therefore use only known characters, and after SFT the model responds to the template
with a (topical, simple) story instead of ignoring the instruction.

Key pieces in `sft.py`: `build_pairs` (synthesize pairs), `encode_pairs` (tokenize into
`(ids, loss_mask)`), `sft_loss` (masked cross-entropy: `(ce * mask).sum() / mask.sum()`), and a
`main` that loads the base `ckpt.npz`, fine-tunes with a small LR for `sft_iters` steps (same
compiled-step machinery as `train.py`), and saves `ckpt_sft.npz` + `ckpt_sft.json`. Sample it
with `python sample.py --chat --ckpt ckpt_sft.npz --meta ckpt_sft.json`.

> **Reality check.** A ~2.7M-param model is far too small to become a real assistant — SFT here
> demonstrates the *mechanism* (template + loss masking + instruction-following emerging), not
> capability. Preference tuning (DPO/RLHF), the stage after SFT, is not implemented; it aligns
> an already-capable model to human preferences and needs a base model far larger than this.

## 7. Configuration reference (`config.py`)

Every hyperparameter lives in one `Config` dataclass. Changing model-shape numbers is the whole
story of LLM scaling in miniature.

### Model shape

| field | default | meaning |
|-------|---------|---------|
| `block_size` | 128 | context length — how many previous tokens the model can see |
| `n_embd` | 192 | model width `C` (size of each token vector); must be divisible by `n_head` |
| `n_head` | 6 | number of attention heads |
| `n_kv_head` | 6 | number of K/V heads; `< n_head` (and dividing it) enables GQA |
| `n_layer` | 6 | number of transformer blocks (depth = reasoning steps) |
| `dropout` | 0.1 | regularization |
| `rope_base` | 10000.0 | RoPE frequency base (θ) |
| `use_qk_norm` | False | RMSNorm Q and K per head before scoring (§3.3) — stability win |
| `use_softmax1` | False | softmax-off-by-one: let attention rows sum to <1 (§3.3) |
| `dtype` | "float32" | weight/compute precision; `"bfloat16"` ~halves memory to fit a bigger model (§4) |
| `use_grad_checkpoint` | False | recompute block activations in backward to save memory (§4) |

> **`config.medium()`** returns a ~20-25M-param preset (`n_embd=512`, `n_layer=8`, `block_size=256`,
> bf16 + grad-checkpoint + QK-Norm). Swap the last line of `config.py` to `config = medium()` to
> train it; expect it to take much longer than the default ~7 min.

### Mixture of experts

| field | default | meaning |
|-------|---------|---------|
| `use_moe` | False | replace each block's FFN with a sparse MoE |
| `n_experts` | 4 | number of expert networks per block |
| `n_experts_per_tok` | 2 | experts each token is routed to (top-k) |
| `moe_aux_coef` | 0.01 | weight of the load-balancing aux loss |

### Training

| field | default | meaning |
|-------|---------|---------|
| `batch_size` | 64 | text windows per gradient step |
| `learning_rate` | 3e-4 | peak LR after warmup |
| `min_lr` | 3e-5 | LR floor at the end of cosine decay |
| `warmup_iters` | 100 | steps to linearly ramp LR from 0 |
| `max_iters` | 5000 | total training steps |
| `eval_interval` | 250 | how often to measure val loss |
| `eval_iters` | 100 | batches averaged per loss estimate |
| `weight_decay` | 0.1 | decoupled decay, weight matrices only |
| `grad_clip` | 1.0 | max gradient norm |

### Entropy-based sampling (`--entropy`)

Used only by `sample.py --entropy` (see §5); all in nats.

| field | default | meaning |
|-------|---------|---------|
| `ent_low` | 0.6 | below this entropy **and** `vent_low` varentropy → greedy argmax |
| `vent_low` | 0.6 | varentropy threshold for the confident/greedy regime |
| `ent_base_temp` | 0.6 | temperature floor when sampling (non-greedy regime) |
| `ent_ent_coef` | 0.3 | how much each nat of entropy heats the temperature |
| `ent_vent_coef` | 0.3 | how much each nat of varentropy heats the temperature |

### Tokenizer & data

| field | default | meaning |
|-------|---------|---------|
| `tokenizer` | "char" | `"char"` or `"bpe"` |
| `bpe_vocab_size` | 1024 | target vocab when using BPE |
| `data_url` | Tiny Shakespeare | corpus download URL |
| `data_path` | shakespeare.txt | local corpus cache |

### Bookkeeping

| field | default | meaning |
|-------|---------|---------|
| `ckpt_path` | ckpt.npz | trained weights |
| `meta_path` | ckpt.json | tokenizer + config sidecar |
| `seed` | 1337 | RNG seed |

---

## 8. Checkpoint format

A checkpoint is **two files** (both gitignored, both regenerable by retraining):

- **`ckpt.npz`** — all weights, saved via `mx.savez` (a flat dict of `param_name → array`).
- **`ckpt.json`** — a human-readable sidecar with two keys:
  - `"tokenizer"` — `tokenizer.to_meta()`: for char, the vocab as one string in id order; for
    BPE, the ordered merge list. `data.load_tokenizer` dispatches on its `type`.
  - `"config"` — the full `Config` as a dict, so `sample.py` rebuilds the exact model shape.

This lets sampling reconstruct both the tokenizer and the model without re-reading the corpus.
**Preserve this contract if you touch save/load.** (Note: weights from the older GPT-2-style
architecture are not loadable by the current model — retrain.)

---

## 9. Parameter budget

The defaults give **~2.68M** parameters. Rough breakdown per the default config
(`n_embd=192`, `n_layer=6`, `n_head=6`, vocab≈65 for char):

- Token embedding (also the tied LM head): `vocab × n_embd`.
- Per block: attention `q_proj + kv_proj + proj` (≈ `3 × n_embd²`) + SwiGLU `gate + up + down`
  (≈ `3 × n_embd × hidden`, `hidden ≈ 8/3·n_embd`) + two RMSNorm gains.
- Final RMSNorm.

Notable effects of the optional features:

- **GQA** (`n_kv_head < n_head`) shrinks `kv_proj` → fewer params (e.g. `n_kv_head=2` → ~2.39M).
- **MoE** multiplies the feed-forward parameters by `n_experts` (e.g. 4 experts → ~8.0M params)
  while keeping per-token compute near `n_experts_per_tok` experts.

If you change `n_embd` / `n_layer` / `n_head`, update the param-count mentions in `README.md`,
`config.py`, and this file.

---

## 10. What's modern vs. GPT-2

`myllm` started as a faithful GPT-2 (2019) and was modernized to ~2024 spec:

| component | GPT-2 (original) | myllm (now) |
|-----------|------------------|-------------|
| position | learned position table | **RoPE** (rotate Q/K) |
| normalization | LayerNorm | **RMSNorm** |
| feed-forward | GELU MLP | **SwiGLU** (+ optional **MoE**) |
| attention | multi-head | multi-head + optional **GQA** |
| inference | recompute full context each token | **KV cache** (O(T)/token) |
| training | constant LR, uniform decay | **warmup→cosine LR**, selective decay, `mx.compile` |
| tokenizer | BPE | char (default) or from-scratch **BPE** |

Each change is the prevailing default in current open models (Llama, Mistral, Qwen, …).

---

## 11. Glossary

- **token** — an integer the model actually consumes; one character (char tokenizer) or a
  subword chunk (BPE).
- **embedding** — a learned vector representing a token (or, originally, a position).
- **logits** — raw, unnormalized next-token scores; softmax turns them into probabilities.
- **head** — one of `n_head` parallel attention sub-spaces.
- **residual stream** — the running `x` that each sub-layer adds into (`x += sublayer(x)`).
- **causal mask** — prevents a position from attending to the future it's predicting.
- **KV cache** — stored past Keys/Values that make incremental generation O(T) per token.
- **RoPE / RMSNorm / SwiGLU / GQA / MoE** — see §3.
- **warmup / cosine decay** — the LR schedule shape (§4).
- **perplexity** — `exp(loss)`; intuitively, the model's effective branching factor.

### Further reading

- Karpathy, *nanoGPT* and *minbpe* — the spiritual ancestors of this repo.
- The MLX docs: <https://ml-explore.github.io/mlx/>.

### Possible next steps (not yet built)

- An attention-weight **heatmap** visualizer.
- A hand-written **autograd** engine (`tensor.py`, micrograd-style) to demystify backprop.
- A true sparse MoE dispatch; preference tuning (DPO) after the SFT in §6; a larger model.
