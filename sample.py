"""
sample.py — generate text from a trained checkpoint.

One-shot (generate once and exit):
    python sample.py --prompt "ROMEO:" --tokens 500

Interactive (load the model ONCE, then keep typing prompts):
    python sample.py -i

This loads ckpt.npz (weights) and ckpt.json (vocab + config), rebuilds the tokenizer from the
saved vocab, encodes your prompt to ids, and lets the model extend it one character at a time
(see GPT.generate in model.py). MLX runs on the Apple-Silicon GPU automatically.

Reminder: this is a *character-level* model trained only on Shakespeare. It is a text
*continuer*, not a chatbot — give it the start of something and it keeps writing in that style.
"""
import argparse
import json

import mlx.core as mx

from config import Config
from data import load_tokenizer
from model import GPT


def load_model(ckpt_path: str, meta_path: str):
    """Rebuild the tokenizer + model config from the JSON sidecar, then load the weights once."""
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    # Rebuild whichever tokenizer was used at train time (char vocab or BPE merges).
    tokenizer = load_tokenizer(meta["tokenizer"])

    cfg = Config(**meta["config"])
    model = GPT(cfg, vocab_size=tokenizer.vocab_size)
    model.load_weights(ckpt_path)
    model.eval()
    return model, tokenizer.encode, tokenizer.decode


def stream_generate(model, idx, n: int, decode, on_char, **gen_kwargs):
    """
    Generate `n` characters, calling on_char(text) for each one as it is produced (so the
    caller can print it live). We hand GPT.generate an `on_token` callback so it can surface
    each character the moment it's sampled while reusing its KV cache internally — one forward
    step per new character instead of re-reading the whole prompt every time.
    """
    return model.generate(idx, n, on_token=lambda tok: on_char(decode([tok])), **gen_kwargs)


def run_once(model, encode, decode, args, gen_kwargs):
    start_ids = encode(args.prompt) or encode("\n")
    idx = mx.array(start_ids)[None]
    out = model.generate(idx, max_new_tokens=args.tokens, **gen_kwargs)
    print(decode(out[0].tolist()))


def run_interactive(model, encode, decode, args, gen_kwargs):
    print("interactive mode — type a prompt and press Enter; Ctrl-D or Ctrl-C to quit.")
    print(f"(temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, "
          f"rep_penalty={args.repetition_penalty}, {args.tokens} chars per reply)")
    while True:
        try:
            prompt = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return

        start_ids = encode(prompt) or encode("\n")
        idx = mx.array(start_ids)[None]
        print(prompt, end="", flush=True)                 # echo the seed, then stream the rest
        stream_generate(model, idx, args.tokens, decode,
                        on_char=lambda ch: print(ch, end="", flush=True), **gen_kwargs)
        print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, default="\n", help="text to start generation from")
    p.add_argument("--tokens", type=int, default=500, help="how many characters to generate")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="higher = more random/creative, lower = more confident/repetitive")
    p.add_argument("--top_k", type=int, default=40,
                   help="only sample from the k most likely next chars (0 = disabled)")
    p.add_argument("--top_p", type=float, default=0.0,
                   help="nucleus sampling: keep the smallest set of chars summing to this "
                        "probability (0 = disabled)")
    p.add_argument("--repetition_penalty", type=float, default=1.0,
                   help="dampen chars already generated to avoid loops (1.0 = off, try ~1.2)")
    p.add_argument("-i", "--interactive", action="store_true",
                   help="load the model once and keep prompting in a loop, streaming output")
    p.add_argument("--ckpt", type=str, default="ckpt.npz")
    p.add_argument("--meta", type=str, default="ckpt.json")
    args = p.parse_args()

    model, encode, decode = load_model(args.ckpt, args.meta)
    # Collect the sampling controls once; 0/off values become None so generate() skips them.
    gen_kwargs = dict(
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        top_p=args.top_p if args.top_p > 0 else None,
        repetition_penalty=args.repetition_penalty,
    )

    if args.interactive:
        run_interactive(model, encode, decode, args, gen_kwargs)
    else:
        run_once(model, encode, decode, args, gen_kwargs)


if __name__ == "__main__":
    main()
