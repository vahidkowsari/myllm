"""
train.py — the training loop.

The whole job of training: repeatedly
    1. grab a random batch of text windows,
    2. ask the model to predict the next char at every position,
    3. measure the loss (how wrong it was),
    4. backprop to get gradients,
    5. nudge the weights to be a little less wrong (optimizer step).

In MLX, steps 2-4 are bundled into ONE call. `nn.value_and_grad(model, loss_fn)` returns a
function that runs the forward pass AND computes the gradients of the loss w.r.t. every
trainable parameter — there is no separate `loss.backward()`. Because MLX is lazy, the actual
math only happens when we `mx.eval(...)` after the optimizer has updated the weights.

The training step is wrapped in `mx.compile`, which fuses the whole forward+backward+update
graph into one optimized kernel — a big speedup, and a nice illustration of what MLX's lazy
graph buys you. We also use a warmup→cosine learning-rate schedule and apply weight decay only
to the weight matrices (not norms/embeddings), both standard practice for training transformers.

Run:  python train.py
It saves the trained weights to ckpt.npz and the tokenizer + config to ckpt.json, which
sample.py then loads.
"""
import json
from dataclasses import asdict
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from config import config
from data import load_data, get_batch
from model import GPT


def loss_fn(model, x, y):
    """Forward pass -> scalar cross-entropy loss. This is the function we differentiate."""
    _, loss, _ = model(x, y)        # (logits, loss, kv_caches) — only the loss matters here
    return loss


def estimate_loss(model, train_data, val_data):
    """
    Measure average loss on train and val splits over a few batches. We do this instead of
    trusting a single noisy batch's loss. Watching val loss tells you if you're overfitting.
    """
    model.eval()                              # turn dropout off while measuring
    out = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        total = 0.0
        for _ in range(config.eval_iters):
            x, y = get_batch(data, config.block_size, config.batch_size)
            total += loss_fn(model, x, y).item()
        out[name] = total / config.eval_iters
    model.train()                             # back to training mode (dropout on)
    return out


def build_decay_mask(params):
    """
    Weight decay should pull the big weight MATRICES toward zero, but NOT the 1-D RMSNorm gains
    or biases (shrinking those just hurts) nor the token embedding. We return a flat dict
    {param_name: 1.0 or 0.0} — 1 where decay applies. The rule "2-D and not the embedding"
    cleanly selects exactly the Linear weights.
    """
    return {
        name: (1.0 if (p.ndim == 2 and not name.startswith("token_emb")) else 0.0)
        for name, p in tree_flatten(params)
    }


def main():
    mx.random.seed(config.seed)

    train_data, val_data, tokenizer = load_data()
    model = GPT(config, tokenizer.vocab_size)
    mx.eval(model.parameters())               # actually allocate/init the (lazy) weights
    print(f"model parameters: {model.num_params()/1e6:.2f}M")

    # Learning-rate schedule: linearly WARM UP from 0 over the first `warmup_iters` steps (so
    # early, half-random gradients don't blow things up), then COSINE-DECAY down to `min_lr`
    # over the rest. join_schedules stitches the two together at the warmup boundary.
    warmup = optim.linear_schedule(0.0, config.learning_rate, config.warmup_iters)
    cosine = optim.cosine_decay(config.learning_rate,
                                config.max_iters - config.warmup_iters, config.min_lr)
    lr_schedule = optim.join_schedules([warmup, cosine], [config.warmup_iters])

    # AdamW with its built-in weight_decay turned OFF — we apply decoupled decay ourselves
    # below so we can restrict it to the weight matrices (see build_decay_mask).
    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=0.0)
    decay_mask = build_decay_mask(model.parameters())

    # Returns (loss, grads) in one shot, differentiating w.r.t. the model's parameters.
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # The whole training step, compiled. `mx.compile` traces the lazy graph once and fuses it
    # into a single optimized kernel; capturing model + optimizer state as inputs AND outputs
    # is how MLX knows those arrays are updated in place across calls.
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y):
        loss, grads = loss_and_grad(model, x, y)        # forward + backprop in one call
        grads, _ = optim.clip_grad_norm(grads, config.grad_clip)   # cap gradient spikes
        optimizer.update(model, grads)                  # nudge every weight against its gradient
        # Decoupled weight decay (the "W" in AdamW): pull weights gently toward zero, but only
        # where the mask is 1 (the weight matrices).  w <- w - lr * weight_decay * w
        lr = optimizer.learning_rate
        decayed = [(name, w - lr * config.weight_decay * decay_mask[name] * w)
                   for name, w in tree_flatten(model.parameters())]
        model.update(tree_unflatten(decayed))
        return loss

    for it in range(config.max_iters + 1):
        # Periodically report how we're doing on both splits. The leading "\r" overwrites the
        # live status line below, and the trailing spaces wipe any leftover characters, so each
        # eval lands cleanly on its own row.
        if it % config.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data)
            print(f"\riter {it:5d} | train loss {losses['train']:.4f} | "
                  f"val loss {losses['val']:.4f}            ")

        # --- one training step ------------------------------------------------
        x, y = get_batch(train_data, config.block_size, config.batch_size)
        loss = step(x, y)
        # Nothing above has actually run yet (MLX is lazy). This forces the computation and
        # materializes the new weights + optimizer state (and the loss, for the status line).
        mx.eval(state)

        # Live status, redrawn in place every iteration (carriage return, no newline). We show
        # the current learning rate too so you can watch warmup ramp it up then cosine ease off.
        pct = 100 * it / config.max_iters
        print(f"\r  {it:5d}/{config.max_iters} ({pct:4.1f}%) | batch loss {loss.item():.4f} | "
              f"lr {optimizer.learning_rate.item():.2e}", end="", flush=True)

    print()   # finish the live status line so the next print starts on a fresh row

    # Save weights to .npz, and the tokenizer + config to a JSON sidecar so sampling can rebuild
    # the tokenizer and the exact model shape. `tokenizer.to_meta()` serializes whichever
    # tokenizer was used (the char vocab string, or the BPE merge list).
    weights = dict(tree_flatten(model.parameters()))
    mx.savez(config.ckpt_path, **weights)
    meta = {
        "tokenizer": tokenizer.to_meta(),
        "config": asdict(config),
    }
    with open(config.meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"saved checkpoint -> {config.ckpt_path} (+ {config.meta_path})")


if __name__ == "__main__":
    main()
