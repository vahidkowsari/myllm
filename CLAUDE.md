# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`myllm` is an **educational** GPT in Apple's **MLX** — a small, heavily-commented nanoGPT-style
transformer whose purpose is *understanding the lowest layers of an LLM*, not production use.
Optimize every change for **readability and pedagogy over performance or cleverness.** The
comments are the product; keep them accurate and plain.

Architecturally it's a ~2024-era transformer: **RoPE** position encoding, **RMSNorm**,
**SwiGLU** feed-forward, a **KV cache** at inference, and config-gated **GQA** (grouped-query
attention) and **MoE** (mixture of experts). Tokenizer is char-level by default with an
optional from-scratch **BPE** (`bpe.py`). Training uses `mx.compile`, a warmup→cosine LR
schedule, and decoupled weight decay on matrices only.

## Environment

- Python **3.13** (NOT 3.14 — MLX has no 3.14 wheels yet). The venv lives in `.venv/`.
- Activate with `source .venv/bin/activate`. Deps: `mlx`, `numpy` (see `requirements.txt`).
- To recreate the env: `./setup.sh` (override interpreter with `PY=python3.x ./setup.sh`).
- **Apple-Silicon only.** MLX runs on the Mac GPU via unified memory — there is no device
  selection (`.to(device)` / `pick_device()` are gone) and no CUDA/CPU fallback.
- MLX is **lazy**: ops build a graph; work happens at `mx.eval(...)` (or `.item()`/`print`).

## Commands

```bash
python data.py                       # download + tokenize the corpus, print stats
python bpe.py                        # standalone BPE demo (train + encode/decode)
python train.py                      # pretrain (5000 iters, ~7 min), saves ckpt.npz + ckpt.json
python sft.py                        # post-train: instruction-tune the base ckpt -> ckpt_sft.*
python sample.py --prompt "ROMEO:"   # one-shot generation from the base checkpoint
python sample.py -i                  # interactive: load once, loop prompts, stream output
python sample.py --chat --ckpt ckpt_sft.npz --meta ckpt_sft.json --prompt "Write a story about a cat."
python vision.py                     # multimodal demo: show synthetic shape images + captions
python train_mm.py                   # train the image->caption model -> ckpt_mm.npz + ckpt_mm.json
python sample_mm.py --n 8            # caption fresh random shape images (rendered in the terminal)
```

`sample.py` flags: `--prompt`, `--tokens`, `--temperature` (0.8), `--top_k` (40), `--top_p`
(0=off), `--repetition_penalty` (1.0=off), `--entropy` (entropy-based "entropix" sampling;
overrides temperature/top_k/top_p), `-i`/`--interactive`, `--chat` (instruction mode,
use with an SFT ckpt), `--ckpt`, `--meta`.

Toggle behavior from `config.py`: `tokenizer` (`"char"`/`"bpe"`) + `bpe_vocab_size`,
`use_moe`/`n_experts`/`n_experts_per_tok`, `n_kv_head` (< `n_head` = GQA),
`use_qk_norm` (RMSNorm Q/K before scoring), `use_softmax1` (softmax-off-by-one attention),
`ent_*`/`vent_*` (entropy-sampling knobs), `data_url`.

Full end-to-end reference (math, every module, config table, training/sampling internals):
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Keep it in sync when you change architecture or
config. A reveal.js slide deck lives at `docs/presentation.html`.

## Layout (read in this order to understand it)

| file        | role |
|-------------|------|
| `data.py`   | tokenizers (`CharTokenizer`, dispatch to `bpe`) + `get_batch` window sampler + save/load. |
| `bpe.py`    | from-scratch byte-level BPE tokenizer (minbpe-style). Used when `config.tokenizer="bpe"`. |
| `model.py`  | the GPT. **Core file.** `CausalSelfAttention` (RoPE/GQA) → `MLP`/`MoE` → `Block` → `GPT`. |
| `train.py`  | pretraining loop (`mx.compile`, warmup→cosine LR, masked weight decay), `estimate_loss`, save. |
| `sft.py`    | post-training: instruction-tune the base ckpt with **loss-masked** (instruction,response) pairs synthesized from the corpus. |
| `sample.py` | autoregressive generation (top-k/top-p/rep-penalty); `-i` REPL, `--chat` instruction mode. |
| `vision.py` | **multimodal demo.** synthetic colored-shape images + `PatchEmbed` (image→vectors) + `CaptionModel` (PatchEmbed + the *unmodified* GPT) + an ANSI-color terminal previewer. |
| `train_mm.py`| train `vision.CaptionModel` image→caption from scratch with loss-masked captions (same masking idea as `sft.py`). |
| `sample_mm.py`| caption fresh random images from `ckpt_mm.*`, rendered in the terminal. |
| `config.py` | single `Config` dataclass with every hyperparameter, commented. |

## Conventions / things to preserve

- **Attention is written out longhand** in `CausalSelfAttention.__call__` (explicit Q·Kᵀ/√d,
  causal mask, softmax, RoPE) *on purpose* — do NOT replace it with
  `mx.fast.scaled_dot_product_attention`. Visibility is the point.
- Tensor shapes are documented inline as `(B, T, C)` (batch, time, channels=n_embd). Keep this
  notation when adding code.
- All hyperparameters go in `config.py` — don't hardcode them elsewhere.
- The checkpoint is **two files**: `ckpt.npz` (weights, via `mx.savez`) + `ckpt.json`
  (`tokenizer.to_meta()` — the char vocab string OR the BPE merge list — plus the `config`) so
  `sample.py` can rebuild the tokenizer and model shape without re-reading the corpus. Tokenizers
  expose `to_meta()` / `from_meta()`; `data.load_tokenizer()` dispatches on the saved `type`.
  Preserve that contract if you touch save/load.
- **Multimodality is one hook, on purpose.** `GPT.__call__` takes an optional `prefix` of
  pre-embedded `(B, T_prefix, C)` vectors that are concatenated in front of the text vectors —
  the model is deliberately *source-agnostic* about where vectors come from. The vision demo
  produces that prefix from image patches (`vision.PatchEmbed`); **do not** push image-specific
  logic into `model.py`. Don't combine `prefix` with `targets` in one call — multimodal loss is
  computed by the caller (`train_mm.caption_loss`), masked to caption tokens.
- The output LM head is **tied** to `token_emb` via `token_emb.as_linear(x)` in `GPT.__call__`
  (MLX's weight-tying idiom) — there is deliberately no separate `lm_head` weight to keep in sync.
- Defaults give a **~2.7M-param** model. If you change `n_embd`/`n_layer`/`n_head`, update the
  param-count mentions in `README.md` and `config.py` to stay accurate.

## Generated/ignored files (in `.gitignore`, safe to delete & regenerate)

`.venv/`, `__pycache__/`, `shakespeare.txt` (re-downloads), `ckpt.npz` + `ckpt.json` (retrain).

## When adding features

Likely next steps the user has discussed: an attention-weight heatmap visualizer, and a
hand-written autograd engine (`tensor.py`, micrograd-style) to demystify backprop. Match the
existing heavy-comment style — assume the reader is learning how LLMs work from this code.
