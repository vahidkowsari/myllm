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

    # --- attention tweaks (recent-paper experiments) -------------------------
    # use_qk_norm: RMSNorm each attention head's Query and Key vectors before computing Q·Kᵀ
    # (the "QK-Norm" trick, now standard in many 2024-25 models). It bounds the size of the
    # attention logits, which stops them blowing up and lets you train at a higher learning
    # rate without divergence. Cheap stability win; off by default to keep the base model plain.
    use_qk_norm: bool = False
    # use_softmax1: the "softmax-off-by-one" / quiet-attention tweak — add a phantom +1 to the
    # softmax denominator so every attention row is free to sum to LESS than 1, i.e. a token can
    # attend to "nothing". This drains the huge outlier activations that otherwise appear in a
    # few channels (which makes the model easier to quantize) and is how "attention sinks" form.
    use_softmax1: bool = False

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

    # --- entropy-based ("entropix") sampling --------------------------------
    # An alternative to fixed temperature/top-k/top-p (sample.py --entropy). At each step we
    # measure the model's OWN uncertainty from the next-token distribution — its entropy (how
    # spread out it is) and varentropy (how spread out the *surprisal* is, i.e. is it torn
    # between a few sharp options or genuinely vague) — and adapt sampling to it: when the model
    # is confident we cool toward greedy; when it is uncertain we heat up. Makes uncertainty
    # visible and audible in the output. Thresholds/coeffs are in nats. See GPT._entropy_sample.
    ent_low: float = 0.6        # below this entropy AND vent_low varentropy -> just take argmax
    vent_low: float = 0.6
    ent_base_temp: float = 0.6  # temperature floor used when sampling (not in the greedy regime)
    ent_ent_coef: float = 0.3   # how much each nat of entropy heats the temperature
    ent_vent_coef: float = 0.3  # how much each nat of varentropy heats the temperature

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

    # --- supervised fine-tuning (SFT / instruction tuning) ------------------
    # After pretraining (train.py), SFT teaches the base model to FOLLOW INSTRUCTIONS instead
    # of just continuing text. We build instruction/response pairs from the story corpus, wrap
    # them in a fixed template, and train only on the RESPONSE tokens (loss masking). See sft.py.
    sft_ckpt_path: str = "ckpt_sft.npz"     # fine-tuned weights (separate from the base ckpt)
    sft_meta_path: str = "ckpt_sft.json"
    sft_lr: float = 1e-4                     # smaller LR than pretraining — just a gentle nudge
    sft_iters: int = 2000                   # SFT needs far fewer steps than pretraining
    sft_warmup_iters: int = 50
    sft_max_pairs: int = 5000               # how many stories to turn into instruction pairs

    # --- multimodal vision demo (vision.py / train_mm.py / sample_mm.py) -----
    # A tiny IMAGE -> TEXT captioning demo bolted onto the SAME GPT. Images of simple colored
    # shapes are chopped into square PATCHES; each patch is projected to an n_embd vector
    # (PatchEmbed, the image analogue of the token embedding) and PREPENDED to the text tokens so
    # the transformer attends to image and text in ONE shared sequence. Loss is graded only on the
    # caption tokens — the exact same masking idea SFT uses. This is the whole "multimodal LLM"
    # trick in miniature: the blocks don't know some of their input vectors came from pixels.
    img_size: int = 24          # square image side in pixels (must be divisible by patch_size)
    patch_size: int = 8         # side of each square patch; (img_size/patch_size)^2 = #image "tokens"
    img_channels: int = 3       # RGB, so the shape's COLOR is visible to the model
    caption_len: int = 40       # fixed text length (chars) per example; the captions are short
    vision_ckpt_path: str = "ckpt_mm.npz"   # weights for the captioning model (GPT + PatchEmbed)
    vision_meta_path: str = "ckpt_mm.json"  # tokenizer + config sidecar, like the other ckpts
    vision_lr: float = 3e-4
    vision_iters: int = 1500    # the shape task is easy; this trains from scratch in ~2 min
    vision_warmup_iters: int = 50

    # --- bookkeeping ---------------------------------------------------------
    data_path: str = "tinystories.txt"  # raw training text
    # MLX saves model weights to a .npz; the tokenizer vocab + this config go in a
    # small human-readable .json sidecar (open it to literally see the vocabulary).
    ckpt_path: str = "ckpt.npz"         # trained weights
    meta_path: str = "ckpt.json"        # vocab + config, so sample.py can rebuild both
    seed: int = 1337


# A single shared instance the other files import.
config = Config()
