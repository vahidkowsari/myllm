"""
model.py — the GPT itself. READ THIS FILE FIRST.

This is the entire model, bottom to top, written in Apple's MLX. Shapes are written in
comments using:
    B = batch size
    T = time / sequence length (number of tokens, <= block_size)
    C = channels = n_embd  (the width of the model)

The flow of a forward pass:
    token ids (B,T)
      -> token embedding   (B,T,C)        "what is this token?"
      -> N transformer blocks             tokens mix info + think
      -> final RMSNorm
      -> lm_head (tied to token emb) (B,T,vocab)   a score for every possible next token
      -> (training) cross-entropy loss against the true next token

This is a ~2024-era transformer rather than the original 2019 GPT-2. The differences from a
textbook GPT, and why each is the modern default:
  - **RoPE** (rotary position embeddings) instead of a learned position table. Position is
    injected by *rotating* the Q/K vectors by an angle that depends on the token's position.
  - **RMSNorm** instead of LayerNorm — same idea (rescale each token vector) but cheaper.
  - **SwiGLU** feed-forward instead of a plain GELU MLP — a *gated* activation.
  - **GQA** (grouped-query attention, optional) — fewer K/V heads than Q heads, which shrinks
    the KV cache. With n_kv_head == n_head it's ordinary multi-head attention.
  - **MoE** (mixture of experts, optional) — replace the single feed-forward with several
    experts + a router that picks a couple per token.
  - **KV cache** in generate() — at inference we remember each layer's past Keys/Values so a
    new token costs one forward step over a single token (O(T) per token instead of O(T^2)).

MLX notes (vs PyTorch):
  - There is no `.to(device)`. MLX uses unified memory and runs on the Apple-Silicon GPU
    by default — arrays just live in one place.
  - MLX is *lazy*: ops build a graph and nothing actually computes until you call
    `mx.eval(...)` (or pull a value out with `.item()` / `print`).
  - An `nn.Module` is literally a dict; any `mx.array` attribute is a trainable parameter.
"""
import mlx.core as mx
import mlx.nn as nn

from config import Config


# --- Rotary Position Embeddings (RoPE) -------------------------------------------------------
# Attention by itself is order-blind: shuffle the tokens and the math is unchanged. Something
# has to tell the model *where* each token is. RoPE rotates each token's Query and Key vectors
# by an angle proportional to its position. The clever part: when you later take Q·K, the dot
# product depends only on the *difference* of the two positions — so the model naturally sees
# "how far apart" two tokens are.

def rope_tables(seq_len: int, head_dim: int, base: float):
    """
    Precompute the rotation angles' cos and sin for positions 0..seq_len-1.
    Returns cos, sin each of shape (seq_len, head_dim/2).
    """
    half = head_dim // 2
    # inv_freq[i] = base^(-i/half): geometrically spaced frequencies, 1 down to ~1/base.
    inv_freq = base ** (-mx.arange(0, half, dtype=mx.float32) / half)   # (half,)
    pos = mx.arange(seq_len, dtype=mx.float32)                          # (seq_len,)
    freqs = pos[:, None] * inv_freq[None, :]                            # (seq_len, half) = angle
    return mx.cos(freqs), mx.sin(freqs)


def apply_rope(x, cos, sin):
    """
    Rotate the vectors in `x` by the precomputed angles (a 2-D rotation per dim pair):
        x1' = x1*cos - x2*sin ;  x2' = x2*cos + x1*sin
    x      : (B, n_head, T, head_dim)
    cos/sin: (T, head_dim/2)
    """
    x1, x2 = mx.split(x, 2, axis=-1)            # the two halves, each (B, n_head, T, head_dim/2)
    cos = cos[None, None]                        # -> (1, 1, T, head_dim/2) to broadcast
    sin = sin[None, None]
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def repeat_kv(x, groups: int):
    """
    Grouped-query attention: each K/V head is shared by `groups` query heads. We physically
    repeat each K/V head `groups` times so the per-head attention math below is unchanged.
    x: (B, n_kv_head, T, head_dim) -> (B, n_kv_head*groups, T, head_dim).
    """
    if groups == 1:
        return x
    return mx.repeat(x, groups, axis=1)


class CausalSelfAttention(nn.Module):
    """
    Self-attention: the one mechanism that makes a transformer a transformer.

    Each token produces a Query ("what am I looking for?"), a Key ("what do I offer?"), and a
    Value ("what do I pass on if attended to?"). A token's new representation is a weighted
    average of the Values of all *earlier* tokens, weighted by how well its Query matches each
    Key:
        attention = softmax( Q · Kᵀ / sqrt(head_dim) )   # (T,T) weights, one row per token
        output    = attention · V

    "Causal" = a mask forces each token to only attend to positions <= its own.
    "Multi-head" = we split C into n_head independent sub-spaces and do the above in each.
    "Grouped-query" = there can be FEWER K/V heads than Q heads (n_kv_head <= n_head); the K/V
    heads are shared across groups of query heads, which makes the KV cache smaller.

    Position enters here, via RoPE: we rotate Q and K by each token's position before scoring.

    NOTE: the attention is written out longhand on purpose (explicit Q·Kᵀ/√d, mask, softmax).
    MLX ships a fused `mx.fast.scaled_dot_product_attention`, but seeing the steps is the point.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        assert cfg.n_head % cfg.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        self.head_dim = cfg.n_embd // cfg.n_head
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.groups = cfg.n_head // cfg.n_kv_head     # how many Q heads share one K/V head
        self.rope_base = cfg.rope_base

        # Separate projections (rather than one fused qkv) because Q and K/V can have different
        # head counts under GQA: Q gets n_head heads, K and V get n_kv_head heads each.
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim)
        self.kv_proj = nn.Linear(cfg.n_embd, 2 * cfg.n_kv_head * self.head_dim)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)   # output projection, after merging heads

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def __call__(self, x, cache=None):
        """
        x     : (B, T, C) the new tokens to process this step.
        cache : optional (past_k, past_v) from earlier steps — the KV cache.
        Returns (output (B,T,C), new_cache).
        """
        B, T, C = x.shape
        hd = self.head_dim

        # Project, then reshape into heads. Q -> (B, n_head, T, hd); K,V -> (B, n_kv_head, T, hd).
        q = self.q_proj(x).reshape(B, T, self.n_head, hd).transpose(0, 2, 1, 3)
        k, v = mx.split(self.kv_proj(x), 2, axis=-1)
        k = k.reshape(B, T, self.n_kv_head, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kv_head, hd).transpose(0, 2, 1, 3)

        # RoPE: rotate Q and K by each token's ABSOLUTE position. `offset` is how many tokens
        # already sit in the cache, so these new tokens get positions offset, offset+1, ...
        offset = 0 if cache is None else cache[0].shape[2]
        cos, sin = rope_tables(offset + T, hd, self.rope_base)
        cos, sin = cos[offset:], sin[offset:]              # the angles for *these* T tokens
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # KV cache: prepend past Keys/Values so each new token attends to the whole past without
        # recomputing it. We cache the SMALL (n_kv_head) tensors — that is the GQA memory win.
        if cache is not None:
            past_k, past_v = cache
            k = mx.concatenate([past_k, k], axis=2)
            v = mx.concatenate([past_v, v], axis=2)
        new_cache = (k, v)
        Tk = k.shape[2]                                    # total keys so far = offset + T

        # Expand the K/V heads to match the Q heads, so the per-head attention below lines up.
        k = repeat_kv(k, self.groups)
        v = repeat_kv(v, self.groups)

        # Attention scores: Q · Kᵀ, scaled by 1/sqrt(head_dim). (B, n_head, T, Tk).
        att = (q @ mx.swapaxes(k, -2, -1)) * (hd ** -0.5)

        # Causal mask, generalized for the cache: query row i is the token at absolute position
        # offset+i; it may attend to key column j only if j <= offset+i. With no cache this is
        # exactly a lower-triangular mask.
        q_pos = offset + mx.arange(T)[:, None]             # (T, 1)
        k_pos = mx.arange(Tk)[None, :]                     # (1, Tk)
        mask = k_pos <= q_pos                              # (T, Tk) True where attending is OK
        att = mx.where(mask, att, float("-inf"))          # block the future with -inf
        att = mx.softmax(att, axis=-1)                    # normalize each row to a distribution
        att = self.attn_dropout(att)

        y = att @ v                                        # (B, n_head, T, hd)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)       # re-merge the heads -> (B, T, C)
        return self.resid_dropout(self.proj(y)), new_cache


class MLP(nn.Module):
    """
    The per-token feed-forward network — a SwiGLU. After attention has let tokens share
    information, this is where each token "thinks" on its own.

    A plain MLP is fc -> GELU -> proj. SwiGLU adds a *gate*: a second linear runs in parallel
    and multiplies the activated branch, so the network can suppress or pass through each
    hidden unit per token:
        hidden = silu(gate(x)) * up(x)          # silu(z) = z * sigmoid(z)
        out    = down(hidden)
    Three matrices instead of two, so we shrink the hidden width to ~8/3·C (Llama's trick) to
    keep the parameter count about the same as a 4·C plain MLP.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        hidden = int(8 * cfg.n_embd / 3)        # e.g. n_embd=192 -> 512; ~ a 4x MLP's param count
        self.gate = nn.Linear(cfg.n_embd, hidden)   # the branch that gets the nonlinearity
        self.up = nn.Linear(cfg.n_embd, hidden)     # the branch that gates it
        self.down = nn.Linear(hidden, cfg.n_embd)   # back down to model width
        self.dropout = nn.Dropout(cfg.dropout)

    def __call__(self, x):
        x = nn.silu(self.gate(x)) * self.up(x)
        return self.dropout(self.down(x))


class MoE(nn.Module):
    """
    A sparse Mixture-of-Experts feed-forward — the drop-in replacement for MLP when
    cfg.use_moe is True.

    Instead of one SwiGLU, we have `n_experts` of them plus a tiny `router` (a single Linear).
    For each token the router scores every expert; we keep the top `k = n_experts_per_tok`,
    softmax those scores into weights, and the token's output is the weighted sum of just those
    experts. So the model holds many experts' worth of *parameters* but each token only "uses"
    a few — capacity without proportional compute. (This is the idea behind Mixtral, etc.)

    Two things make MoE actually work:
      1. top-k routing (here), and
      2. a load-balancing auxiliary loss (returned alongside the output) that punishes the
         router for dumping every token on one expert — otherwise it collapses to a favorite
         and the others never learn.

    PEDAGOGICAL SIMPLIFICATION: for clarity we run *every* token through *every* expert and
    then zero out the ones a token didn't pick. That throws away MoE's compute savings (a real
    implementation gathers only the routed tokens per expert), but it keeps the routing math
    readable and is harmless on a tiny model.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.k = cfg.n_experts_per_tok
        self.router = nn.Linear(cfg.n_embd, cfg.n_experts)        # token -> a score per expert
        self.experts = [MLP(cfg) for _ in range(cfg.n_experts)]   # each expert is a SwiGLU

    def __call__(self, x):
        B, T, C = x.shape
        x = x.reshape(-1, C)                       # (N, C) — treat every token independently
        N = x.shape[0]

        logits = self.router(x)                    # (N, E) router score for each expert
        probs = mx.softmax(logits, axis=-1)        # (N, E) full distribution (used by aux loss)

        # Pick the top-k experts per token and re-softmax just those scores into mixing weights.
        order = mx.argsort(-logits, axis=-1)       # experts sorted best-first, (N, E)
        top_idx = order[:, : self.k]               # (N, k) the chosen experts' ids
        top_logits = mx.take_along_axis(logits, top_idx, axis=-1)   # (N, k)
        top_w = mx.softmax(top_logits, axis=-1)    # (N, k) weights that sum to 1 per token

        # Run each expert over all tokens and add in its contribution, weighted by how much
        # (if at all) each token routed to it.
        out = mx.zeros_like(x)
        for e in range(self.n_experts):
            chosen = (top_idx == e)                                 # (N, k) bool
            weight = mx.sum(mx.where(chosen, top_w, 0.0), axis=-1)  # (N,) 0 if token didn't pick e
            out = out + self.experts[e](x) * weight[:, None]

        # Load-balancing aux loss (Switch-Transformer style): f = fraction of tokens that chose
        # each expert; P = mean router probability for each expert. Their dot product is
        # minimized when both are uniform, i.e. when load is spread evenly.
        membership = mx.max(
            (top_idx[..., None] == mx.arange(self.n_experts)).astype(mx.float32), axis=1
        )                                          # (N, E) 1 if expert e is in the token's top-k
        f = membership.mean(axis=0)                # (E,) fraction of tokens choosing each expert
        P = probs.mean(axis=0)                     # (E,) mean router prob per expert
        aux = self.n_experts * mx.sum(f * P)

        return out.reshape(B, T, C), aux


class Block(nn.Module):
    """
    One transformer block = attention + feed-forward, each preceded by an RMSNorm and wrapped
    in a residual (skip) connection:
        x = x + attention(rmsnorm(x))
        x = x + feedforward(rmsnorm(x))
    The residual `x +` lets gradients flow straight through, so you can stack many blocks.
    RMSNorm keeps each token vector at a sane scale before each sub-layer ("pre-norm").
    The feed-forward is a dense SwiGLU (MLP) by default, or a sparse MoE if cfg.use_moe.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        # RMSNorm: divide each token vector by its root-mean-square, then scale by a learned
        # per-channel gain. Like LayerNorm but with no mean-centering and no bias — cheaper.
        self.ln1 = nn.RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.RMSNorm(cfg.n_embd)
        self.ffn = MoE(cfg) if cfg.use_moe else MLP(cfg)

    def __call__(self, x, cache=None):
        h, cache = self.attn(self.ln1(x), cache)
        x = x + h
        ff = self.ffn(self.ln2(x))
        # MoE returns (output, aux_loss); the dense MLP returns just the output.
        if isinstance(ff, tuple):
            ff, aux = ff
        else:
            ff, aux = ff, mx.array(0.0)
        x = x + ff
        return x, cache, aux


class GPT(nn.Module):
    """The full model: token embedding -> stack of Blocks -> final norm -> vocab logits."""

    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size

        # Token embedding: a lookup table mapping each token id to a vector. There is NO
        # position table — position is supplied by RoPE inside attention.
        self.token_emb = nn.Embedding(vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = [Block(cfg) for _ in range(cfg.n_layer)]
        self.ln_f = nn.RMSNorm(cfg.n_embd)

        # Weight tying: the output projection (the "LM head") REUSES the token embedding matrix
        # via `token_emb.as_linear(x)` in forward — there is deliberately no separate lm_head.
        self._init_weights()

    def _init_weights(self):
        # GPT-2 init recipe (normal, std 0.02). The output projection of each residual sub-layer
        # (attention's `proj` and the feed-forward's `down`) is additionally scaled by
        # 1/sqrt(2*n_layer): with N blocks each adding into the residual stream, this keeps the
        # stream's variance from growing with depth, which stabilizes training. (RMSNorm gains
        # are left at their default 1.0.)
        n_layer = self.cfg.n_layer
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                std = 0.02
                if name.endswith("attn.proj") or name.endswith(".down"):
                    std = 0.02 / (2 * n_layer) ** 0.5
                module.weight = mx.random.normal(module.weight.shape) * std
                if "bias" in module:
                    module.bias = mx.zeros(module.bias.shape)
            elif isinstance(module, nn.Embedding):
                module.weight = mx.random.normal(module.weight.shape) * 0.02

    def num_params(self) -> int:
        # Sum the sizes of every parameter array. token_emb is counted once (it doubles as the
        # output projection via weight tying), so this is the "real" parameter count.
        from mlx.utils import tree_flatten
        return sum(p.size for _, p in tree_flatten(self.parameters()))

    def __call__(self, idx, targets=None, caches=None, prefix=None):
        """
        idx     : (B, T) token ids.
        targets : (B, T) the next-token ids, or None at generation time.
        caches  : optional list of per-block (k, v) caches; None during training.
        prefix  : optional (B, T_prefix, C) block of vectors from ANOTHER modality (e.g. image
                  patches; see vision.py), glued on in front of the text vectors. This is the
                  multimodal hook — see the comment below.
        Returns (logits, loss, new_caches). loss is None if targets is None.
        """
        B, T = idx.shape
        T_prefix = 0 if prefix is None else prefix.shape[1]
        assert T + T_prefix <= self.block_size, \
            f"sequence length {T + T_prefix} exceeds block_size {self.block_size}"

        x = self.token_emb(idx)                            # (B, T, C) — position comes from RoPE
        if prefix is not None:
            # THE MULTIMODAL HOOK. `prefix` is a block of (B, T_prefix, C) vectors produced by some
            # OTHER modality and simply concatenated in front of the text vectors. The blocks below
            # can't tell the difference: to them it is all just a sequence of C-dim vectors, and
            # attention lets the text positions look at these extra vectors exactly as they look at
            # each other. That source-agnostic property is the entire reason a transformer can be
            # made multimodal — nothing inside attention changes. (Used at training / prefill; the
            # loss for these prefix positions is handled by the caller, so don't combine `prefix`
            # with `targets` here.)
            x = mx.concatenate([prefix, x], axis=1)        # (B, T_prefix + T, C)
        x = self.drop(x)

        if caches is None:                                 # training / first step: empty caches
            caches = [None] * len(self.blocks)
        new_caches = []
        aux_total = mx.array(0.0)                          # accumulates MoE load-balancing loss
        for block, cache in zip(self.blocks, caches):      # the deep stack of reasoning
            x, cache, aux = block(x, cache)
            new_caches.append(cache)
            aux_total = aux_total + aux
        x = self.ln_f(x)
        # Tied LM head: (B, T, C) -> (B, T, vocab) next-token scores.
        logits = self.token_emb.as_linear(x)

        loss = None
        if targets is not None:
            # Cross-entropy at every position: how surprised the model was by the true next
            # token, averaged. Plus the MoE aux loss (zero unless use_moe is on).
            loss = nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="mean",
            )
            loss = loss + self.cfg.moe_aux_coef * aux_total
        return logits, loss, new_caches

    # --- sampling helpers --------------------------------------------------------------------

    def _apply_repetition_penalty(self, logits, idx, penalty):
        """
        Discourage the model from looping by dampening the logits of tokens already in `idx`.
        HuggingFace convention: divide positive logits / multiply negative logits by `penalty`
        (>1), which pushes already-seen tokens toward lower probability either way.
        logits: (B, vocab) ; idx: (B, T) the sequence so far.
        """
        vocab = logits.shape[-1]
        # (B, vocab) bool: which token ids have appeared in each row's history.
        seen = (idx[..., None] == mx.arange(vocab)).any(axis=1)
        penalized = mx.where(logits > 0, logits / penalty, logits * penalty)
        return mx.where(seen, penalized, logits)

    def _sample_top_p(self, logits, top_p):
        """
        Nucleus sampling: keep the smallest set of most-likely tokens whose probabilities sum
        to >= top_p, then sample from just those. Adapts the candidate set per step (unlike
        top_k's fixed count). We sort, mask the tail, and sample in sorted space to avoid an
        unsort. logits: (B, vocab) -> next-token ids (B,).
        """
        order = mx.argsort(-logits, axis=-1)               # tokens sorted most-likely first
        sorted_logits = mx.take_along_axis(logits, order, axis=-1)
        probs = mx.softmax(sorted_logits, axis=-1)
        cum_before = mx.cumsum(probs, axis=-1) - probs     # cumulative prob BEFORE each token
        keep = cum_before < top_p                          # keep up to & including the crossing
        sorted_logits = mx.where(keep, sorted_logits, float("-inf"))
        choice = mx.random.categorical(sorted_logits)      # (B,) index into the sorted order
        return mx.take_along_axis(order, choice[:, None], axis=-1)[:, 0]

    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, top_p: float | None = None,
                 repetition_penalty: float = 1.0, on_token=None):
        """
        Autoregressive sampling with a KV cache. Each step:
          1. forward pass -> logits for the next token
          2. (optional) repetition penalty on tokens already generated
          3. scale by temperature
          4. (optional) top_k and/or top_p filtering, then sample one token
          5. append it and repeat
        If `on_token` is given it is called with each new token id as it is produced, so a
        caller can stream output live (sample.py's -i mode uses this). If `on_token` returns a
        truthy value, generation stops early — used by chat mode to halt at a stop marker.

        The KV cache is the speed trick: each block remembers its past Keys/Values, so a new
        token is one forward pass over a *single* token (O(T) instead of O(T^2)). We rebuild
        the cache from scratch only when we don't have one yet, or when the sequence has grown
        past block_size — re-priming on the last block_size tokens keeps every RoPE position
        inside [0, block_size), the range the model was trained on.
        """
        caches = None
        for _ in range(max_new_tokens):
            if caches is None or idx.shape[1] > self.block_size:
                logits, _, caches = self(idx[:, -self.block_size:], caches=None)
            else:
                logits, _, caches = self(idx[:, -1:], caches=caches)

            logits = logits[:, -1, :]                      # (B, vocab) — only the last step
            if repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(logits, idx, repetition_penalty)
            logits = logits / temperature
            if top_k is not None:
                k = min(top_k, logits.shape[-1])
                kth = mx.sort(logits, axis=-1)[:, -k][:, None]      # k-th largest logit per row
                logits = mx.where(logits < kth, float("-inf"), logits)

            if top_p is not None:
                next_id = self._sample_top_p(logits, top_p)
            else:
                next_id = mx.random.categorical(logits)    # sample (not argmax) for variety
            next_id = next_id[:, None].astype(idx.dtype)

            idx = mx.concatenate([idx, next_id], axis=1)
            mx.eval(idx)
            if on_token is not None and on_token(int(next_id[0, 0])):
                break                                      # callback asked to stop (e.g. EOT)
        return idx
