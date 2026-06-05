"""
config.py — every knob in one place.

These defaults make a *small* model (~2.7M params) that trains in a few minutes on a laptop
on Apple-Silicon (MLX runs on the GPU via unified memory — no device juggling needed).
Raise the model-size numbers to make it smarter (and slower). That trade-off is the whole
story of LLM scaling, in miniature.
"""
from dataclasses import dataclass


@dataclass
class Config:
    # --- model shape ---------------------------------------------------------
    # block_size = context length = how many previous characters the model can
    # look at when predicting the next one.
    block_size: int = 128
    # n_embd = the width of the model: the size of each token's vector. Every
    # embedding, attention projection, and MLP works in this many dimensions.
    n_embd: int = 192
    # n_head = how many parallel attention "heads". n_embd must be divisible by it.
    n_head: int = 6
    # n_kv_head = number of KEY/VALUE heads. With n_kv_head == n_head this is ordinary
    # multi-head attention. Set it LOWER (must divide n_head) for Grouped-Query Attention:
    # several query heads share one K/V head, shrinking the KV cache at inference for a small
    # quality cost. Llama-2-70B, Mistral, Qwen, etc. all do this. Default = plain MHA.
    n_kv_head: int = 6
    # n_layer = how many transformer blocks are stacked. Depth = reasoning steps.
    n_layer: int = 6
    # dropout = regularization; 0.0 is fine for this tiny dataset.
    dropout: float = 0.1
    # rope_base = the "theta" of Rotary Position Embeddings. Positions are encoded by
    # *rotating* each token's Q/K vectors by an angle proportional to its position (there is
    # no learned position table anymore). This base sets how fast the rotation frequencies
    # fall off across the head dimension; 10000 is the value used by GPT-NeoX/Llama/etc.
    rope_base: float = 10000.0

    # --- mixture of experts (MoE) -------------------------------------------
    # When use_moe is True, each block's feed-forward becomes a SPARSE mixture of experts:
    # `n_experts` separate SwiGLU networks plus a tiny router that sends each token to its top
    # `n_experts_per_tok`. You get many experts' worth of parameters but pay (roughly) the
    # compute of only a few per token. Off by default — the dense SwiGLU is simpler to read.
    use_moe: bool = False
    n_experts: int = 4
    n_experts_per_tok: int = 2
    # Load-balancing aux loss weight: nudges the router to spread tokens across all experts
    # instead of collapsing onto one favorite. Added to the training loss only when use_moe.
    moe_aux_coef: float = 0.01

    # --- training ------------------------------------------------------------
    batch_size: int = 64        # how many text chunks per gradient step
    learning_rate: float = 3e-4 # peak AdamW step size (after warmup)
    min_lr: float = 3e-5        # cosine-decay the LR down to this by the final step
    warmup_iters: int = 100     # linearly ramp the LR up from 0 over the first N steps
    max_iters: int = 5000       # total training steps
    eval_interval: int = 250    # how often to measure val loss
    eval_iters: int = 100       # how many batches to average for each loss estimate
    weight_decay: float = 0.1   # decoupled weight decay, applied to weight matrices only
    grad_clip: float = 1.0      # clip gradients to this norm (training stability)

    # --- tokenizer & data ----------------------------------------------------
    # "char" = one token per character (zero magic, tiny vocab). "bpe" = a small subword
    # tokenizer trained on the corpus (see bpe.py): one token ~ a few characters, so the same
    # block_size covers far more text. char is the default for transparency.
    tokenizer: str = "char"
    bpe_vocab_size: int = 1024          # target vocab size when tokenizer == "bpe"
    # Where the raw training text comes from. Swap this (and data_path) to train on something
    # else. Currently: TinyStories (valid split, ~22MB) — simple synthetic kids' stories that
    # a tiny model can actually learn to write coherently. The original Tiny Shakespeare lives
    # at https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    data_url: str = (
        "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
        "TinyStoriesV2-GPT4-valid.txt"
    )

    # --- bookkeeping ---------------------------------------------------------
    data_path: str = "tinystories.txt"  # raw training text
    # MLX saves model weights to a .npz; the tokenizer vocab + this config go in a
    # small human-readable .json sidecar (open it to literally see the vocabulary).
    ckpt_path: str = "ckpt.npz"         # trained weights
    meta_path: str = "ckpt.json"        # vocab + config, so sample.py can rebuild both
    seed: int = 1337


# A single shared instance the other files import.
config = Config()
