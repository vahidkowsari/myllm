"""
train_mm.py — train the multimodal (image -> caption) model from scratch.

This is train.py + sft.py's masked loss, applied to the CaptionModel in vision.py. The recipe:

  1. Each step, draw a batch of random (image, caption) pairs (vision.make_batch).
  2. Embed the image into patch vectors, PREPEND them to the "Caption: " text, run the GPT.
  3. Cross-entropy on the predicted characters, but MASKED so only the caption tokens count
     (we are not trying to predict pixels or the fixed lead-in) — same trick as sft.py.
  4. Backprop and step. The PatchEmbed and the GPT are trained TOGETHER, so the image encoder
     learns to produce vectors the text side can actually use. That joint training is what makes
     the two modalities "speak the same language".

The task is deliberately trivial (9 shape/color combos), so a from-scratch ~2.7M model nails it
in a couple of minutes. After training we caption a few fresh, unseen images to prove it learned.

Run:  python train_mm.py
"""
import json
from dataclasses import asdict
from functools import partial

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from config import config
from vision import (CaptionModel, build_caption_tokenizer, make_batch, make_image,
                    ascii_image, caption_for, SHAPES, COLOR_NAMES)


def caption_loss(model, images, x, y, m):
    """
    Cross-entropy over the caption characters only. The model's logits cover the whole sequence
    (image patches + text); we slice off the image-patch positions so what's left lines up with
    the text targets y, then weight by the mask m (1 on caption tokens). Identical in spirit to
    sft.py's sft_loss — the prefix slice here plays the role of SFT's instruction mask.
    """
    logits = model(images, x)                              # (B, n_patches + T_text, vocab)
    n_patches = model.patch_embed.n_patches
    text_logits = logits[:, n_patches:, :]                 # drop the image positions -> (B, T_text, vocab)
    vocab = text_logits.shape[-1]
    ce = nn.losses.cross_entropy(
        text_logits.reshape(-1, vocab), y.reshape(-1), reduction="none"
    ).reshape(y.shape)
    return (ce * m).sum() / (m.sum() + 1e-8)


def main():
    mx.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)

    tokenizer = build_caption_tokenizer()
    model = CaptionModel(config, tokenizer.vocab_size)
    mx.eval(model.parameters())
    print(f"caption model: {sum(p.size for _, p in tree_flatten(model.parameters()))/1e6:.2f}M params "
          f"({model.patch_embed.n_patches} image tokens + up to {config.caption_len} text tokens)")

    # Same training machinery as train.py / sft.py: warmup -> cosine LR, compiled step, weight
    # decay on the weight matrices only. We build the decay mask inline because the params are now
    # nested under "gpt." / "patch_embed." (so the rule is "2-D, and not the tied token embedding").
    warmup = optim.linear_schedule(0.0, config.vision_lr, config.vision_warmup_iters)
    cosine = optim.cosine_decay(config.vision_lr,
                                config.vision_iters - config.vision_warmup_iters, config.vision_lr * 0.1)
    lr_schedule = optim.join_schedules([warmup, cosine], [config.vision_warmup_iters])
    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=0.0)
    decay_mask = {
        name: (1.0 if (p.ndim == 2 and "token_emb" not in name) else 0.0)
        for name, p in tree_flatten(model.parameters())
    }
    loss_and_grad = nn.value_and_grad(model, caption_loss)
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(images, x, y, m):
        loss, grads = loss_and_grad(model, images, x, y, m)
        grads, _ = optim.clip_grad_norm(grads, config.grad_clip)
        optimizer.update(model, grads)
        lr = optimizer.learning_rate
        decayed = [(name, w - lr * config.weight_decay * decay_mask[name] * w)
                   for name, w in tree_flatten(model.parameters())]
        model.update(tree_unflatten(decayed))
        return loss

    model.train()
    for it in range(config.vision_iters + 1):
        images, x, y, m = make_batch(config.batch_size, tokenizer.encode, config, rng)
        loss = step(images, x, y, m)
        mx.eval(state)
        if it % 100 == 0:
            print(f"\riter {it:5d}/{config.vision_iters} | loss {loss.item():.4f} | "
                  f"lr {optimizer.learning_rate.item():.2e}            ")
        else:
            print(f"\r  {it:5d}/{config.vision_iters} | loss {loss.item():.4f}", end="", flush=True)
    print()

    # Save weights + the tokenizer/config sidecar, mirroring the other checkpoints so sample_mm.py
    # can rebuild the exact tokenizer and model shape.
    weights = dict(tree_flatten(model.parameters()))
    mx.savez(config.vision_ckpt_path, **weights)
    with open(config.vision_meta_path, "w", encoding="utf-8") as f:
        json.dump({"tokenizer": tokenizer.to_meta(), "config": asdict(config)},
                  f, ensure_ascii=False, indent=2)
    print(f"saved checkpoint -> {config.vision_ckpt_path} (+ {config.vision_meta_path})\n")

    # Payoff: caption a few FRESH images the model never saw during this step.
    model.eval()
    print("captioning fresh images:\n")
    for _ in range(4):
        shape = SHAPES[int(rng.integers(len(SHAPES)))]
        color = COLOR_NAMES[int(rng.integers(len(COLOR_NAMES)))]
        img = make_image(shape, color, config, rng)
        pred = model.caption(img, tokenizer.encode, tokenizer.decode)
        print(ascii_image(img))
        print(f"  true: {caption_for(shape, color)}")
        print(f"  pred: {pred}\n")


if __name__ == "__main__":
    main()
