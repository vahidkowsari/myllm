"""
sft.py — Supervised Fine-Tuning (a.k.a. instruction tuning), the first stage of "post-training".

Pretraining (train.py) gives you a *base* model: a pure text continuer. Give it
"Write a story about a dog." and it won't obey — it'll just continue that sentence, because all
it ever learned is "predict the next character of story-text".

SFT teaches it to FOLLOW INSTRUCTIONS. The recipe is only two changes from pretraining:

  1. DATA SHAPE. Instead of raw text, we use (instruction, response) pairs wrapped in a fixed
     template the model can learn to recognize:

         Instruction:
         {instruction}

         Response:
         {response}<|endoftext|>

  2. LOSS MASKING. We only grade the model on the RESPONSE tokens. It should learn to *produce*
     the response, not to parrot back the instruction. So the loss for every token in the
     "Instruction: …" part (and the "Response:" header) is masked to zero.

We don't have a human-written instruction dataset, so we SYNTHESIZE one from the same
TinyStories corpus: split it into individual stories, pick a topic word from each, and make the
instruction "Write a story about {topic}." with the story as the response. The pairs are
therefore guaranteed to use only characters the base model already knows.

HONEST CAVEAT: a ~2.7M-param model is far too small to become a real assistant. The point here
is to *see the mechanism* — after SFT, the model responds to the instruction template with a
(topical, simple) story instead of ignoring it. That is instruction-following, in miniature.

Run:  python sft.py            (after train.py has produced ckpt.npz)
Then: python sample.py --chat --ckpt ckpt_sft.npz --meta ckpt_sft.json --prompt "Write a story about a cat."
"""
import json
import re
from collections import Counter
from dataclasses import asdict
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from config import config
from data import load_tokenizer, download_if_needed
from model import GPT
from train import build_decay_mask

# The instruction template. Uses ONLY characters in the TinyStories vocab (no '#'), and ends the
# response with the same "<|endoftext|>" marker the base model already saw between stories — so
# it doubles as a natural "stop generating" signal. PROMPT_TEMPLATE is shared with sample.py's
# chat mode so training and inference format prompts identically.
PROMPT_TEMPLATE = "Instruction:\n{instruction}\n\nResponse:\n"
EOT = "<|endoftext|>"

# Common words to skip when guessing a story's topic, so we pick a content word.
_STOP = set("a an the and or but to of in on at is was were are be it he she they him her his "
            "with for that this then so as had has have you i we my me not no yes one day there "
            "their them when who what then once upon time".split())


def _topic(story: str) -> str:
    """Heuristically pick what a story is 'about': its most frequent content word."""
    words = [w for w in re.findall(r"[a-z]{3,}", story.lower()) if w not in _STOP]
    return Counter(words).most_common(1)[0][0] if words else "something"


def build_pairs(text: str, max_pairs: int):
    """Turn the raw corpus into a list of (instruction, response) string pairs."""
    # A few phrasings so the model learns the *pattern*, not one exact sentence. We pick the
    # phrasing by story index (no randomness — keeps runs reproducible).
    phrasings = [
        "Write a story about {t}.",
        "Tell me a story about {t}.",
        "Can you write a short story about {t}?",
        "Write a little story about {t}.",
    ]
    pairs = []
    for i, story in enumerate(s.strip() for s in text.split(EOT)):
        # Keep tidy, complete-looking stories: a sensible length, starting with a capital letter.
        if not (150 <= len(story) <= 600) or not story[:1].isupper():
            continue
        t = _topic(story)
        pairs.append((phrasings[i % len(phrasings)].format(t=t), story))
        if len(pairs) >= max_pairs:
            break
    return pairs


def encode_pairs(pairs, encode, block_size):
    """
    Tokenize each pair into (ids, loss_mask), both length block_size+1 (so we can shift by one
    for next-token prediction). loss_mask is 1 on RESPONSE tokens (incl. the EOT marker) and 0
    on the instruction/prompt tokens — that is the whole trick that makes SFT 'supervised'.
    """
    pad = 0                                              # padding id; masked out, so its value is moot
    examples = []
    for instruction, response in pairs:
        prompt = PROMPT_TEMPLATE.format(instruction=instruction)
        prompt_ids = encode(prompt)
        resp_ids = encode(response + EOT)
        ids = prompt_ids + resp_ids
        mask = [0] * len(prompt_ids) + [1] * len(resp_ids)   # grade only the response
        ids, mask = ids[: block_size + 1], mask[: block_size + 1]
        if sum(mask) == 0:                              # response got truncated away entirely
            continue
        ids += [pad] * (block_size + 1 - len(ids))      # right-pad to a fixed length
        mask += [0] * (block_size + 1 - len(mask))
        examples.append((ids, mask))
    ids = mx.array([e[0] for e in examples])            # (N, block_size+1)
    mask = mx.array([e[1] for e in examples], dtype=mx.float32)
    return ids, mask


def get_sft_batch(ids, mask, batch_size):
    """Sample a batch and split into (x, y, loss_mask) for shifted next-token prediction."""
    n = ids.shape[0]
    pick = mx.random.randint(0, n, shape=(batch_size,))
    b_ids, b_mask = ids[pick], mask[pick]
    x = b_ids[:, :-1]                                   # inputs
    y = b_ids[:, 1:]                                    # the next token at each position
    m = b_mask[:, 1:]                                   # loss weight for predicting each y
    return x, y, m


def sft_loss(model, x, y, m):
    """Cross-entropy, but averaged ONLY over the response tokens (where m == 1)."""
    logits, _, _ = model(x)
    vocab = logits.shape[-1]
    ce = nn.losses.cross_entropy(
        logits.reshape(-1, vocab), y.reshape(-1), reduction="none"
    ).reshape(y.shape)
    return (ce * m).sum() / (m.sum() + 1e-8)


def main():
    mx.random.seed(config.seed)

    # Load the PRETRAINED base model + its tokenizer (the starting point we fine-tune).
    with open(config.meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    tokenizer = load_tokenizer(meta["tokenizer"])
    cfg = type(config)(**meta["config"])               # the config the base model was trained with
    model = GPT(cfg, vocab_size=tokenizer.vocab_size)
    model.load_weights(config.ckpt_path)
    mx.eval(model.parameters())
    print(f"loaded base model: {model.num_params()/1e6:.2f}M params")

    # Build the instruction dataset from the corpus.
    text = download_if_needed()
    pairs = build_pairs(text, config.sft_max_pairs)
    ids, mask = encode_pairs(pairs, tokenizer.encode, cfg.block_size)
    print(f"SFT examples: {ids.shape[0]} pairs "
          f"(avg {float(mask.sum())/ids.shape[0]:.0f} response tokens each)")
    print("example prompt:\n" + PROMPT_TEMPLATE.format(instruction=pairs[0][0]))

    # Same training machinery as train.py: warmup→cosine LR, compiled step, masked weight decay.
    warmup = optim.linear_schedule(0.0, config.sft_lr, config.sft_warmup_iters)
    cosine = optim.cosine_decay(config.sft_lr, config.sft_iters - config.sft_warmup_iters, config.sft_lr * 0.1)
    lr_schedule = optim.join_schedules([warmup, cosine], [config.sft_warmup_iters])
    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=0.0)
    decay_mask = build_decay_mask(model.parameters())
    loss_and_grad = nn.value_and_grad(model, sft_loss)

    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y, m):
        loss, grads = loss_and_grad(model, x, y, m)
        grads, _ = optim.clip_grad_norm(grads, config.grad_clip)
        optimizer.update(model, grads)
        lr = optimizer.learning_rate
        decayed = [(name, w - lr * config.weight_decay * decay_mask[name] * w)
                   for name, w in tree_flatten(model.parameters())]
        model.update(tree_unflatten(decayed))
        return loss

    model.train()
    for it in range(config.sft_iters + 1):
        x, y, m = get_sft_batch(ids, mask, config.batch_size)
        loss = step(x, y, m)
        mx.eval(state)
        if it % 100 == 0:
            print(f"\rsft iter {it:5d}/{config.sft_iters} | loss {loss.item():.4f} | "
                  f"lr {optimizer.learning_rate.item():.2e}            ")
        else:
            print(f"\r  {it:5d}/{config.sft_iters} | loss {loss.item():.4f}", end="", flush=True)
    print()

    # Save the fine-tuned model alongside the SAME tokenizer + config so sample.py can load it.
    weights = dict(tree_flatten(model.parameters()))
    mx.savez(config.sft_ckpt_path, **weights)
    with open(config.sft_meta_path, "w", encoding="utf-8") as f:
        json.dump({"tokenizer": meta["tokenizer"], "config": asdict(cfg)}, f, ensure_ascii=False, indent=2)
    print(f"saved fine-tuned checkpoint -> {config.sft_ckpt_path} (+ {config.sft_meta_path})")


if __name__ == "__main__":
    main()
