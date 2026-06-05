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
import mlx.nn as nn
from mlx.utils import tree_flatten

from config import Config
from data import load_tokenizer
from model import GPT
from sft import PROMPT_TEMPLATE, EOT      # instruction template + stop marker, for --chat mode


def _nbytes(model) -> int:
    """Total bytes of all parameter arrays — used to show the quantization size win."""
    return sum(p.nbytes for _, p in tree_flatten(model.parameters()))


def quantize_model(model, bits: int, group_size: int):
    """
    Shrink the model to `bits`-bit weights in place with MLX's nn.quantize. Each weight matrix is
    split into groups of `group_size` columns; each group is stored as small ints plus a scale
    (and zero-point), so a 4-bit model is ~1/8 the size of float32 and decodes faster.

    We only quantize Linear/Embedding layers whose last dimension divides evenly into the group
    size — that skips tiny oddballs like the MoE router (which would error and isn't worth it).
    The tied LM head comes along for free: it reuses the now-quantized token-embedding weight.
    """
    def can_quantize(_path, module):
        return (isinstance(module, (nn.Linear, nn.Embedding))
                and module.weight.shape[-1] % group_size == 0)
    nn.quantize(model, group_size=group_size, bits=bits, class_predicate=can_quantize)


def load_model(ckpt_path: str, meta_path: str, quantize: int = 0, q_group_size: int = 64):
    """Rebuild the tokenizer + model config from the JSON sidecar, then load the weights once."""
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    # Rebuild whichever tokenizer was used at train time (char vocab or BPE merges).
    tokenizer = load_tokenizer(meta["tokenizer"])

    cfg = Config(**meta["config"])
    model = GPT(cfg, vocab_size=tokenizer.vocab_size)
    model.load_weights(ckpt_path)
    if quantize:
        before = _nbytes(model)
        quantize_model(model, bits=quantize, group_size=q_group_size)
        mx.eval(model.parameters())
        after = _nbytes(model)
        print(f"quantized to {quantize}-bit (group {q_group_size}): "
              f"{before/1e6:.1f}MB -> {after/1e6:.1f}MB ({before/after:.1f}x smaller)")
    model.eval()
    return model, tokenizer.encode, tokenizer.decode


class StopStreamer:
    """
    Streams generated characters to `emit`, but stops at a `stop` string and never prints it.
    To avoid leaking a partial marker, it holds back the last len(stop)-1 characters until they
    are proven not to be the start of the marker. Used in --chat mode to cut the model off at
    "<|endoftext|>". `feed` returns True once the marker is seen (GPT.generate then stops).
    """

    def __init__(self, stop, emit):
        self.stop, self.emit, self.buf = stop, emit, ""

    def feed(self, ch):
        self.buf += ch
        hit = self.buf.find(self.stop)
        if hit != -1:
            self.emit(self.buf[:hit])                      # emit text before the marker, then stop
            self.buf = ""
            return True
        keep = len(self.stop) - 1                          # might be the start of the marker
        if len(self.buf) > keep:
            self.emit(self.buf[:-keep])
            self.buf = self.buf[-keep:]
        return False

    def close(self):
        if self.buf:                                       # flush leftovers if no marker appeared
            self.emit(self.buf)
            self.buf = ""


def generate_reply(model, encode, decode, user_prompt, args, gen_kwargs, on_char):
    """
    Generate one reply, streaming each character to `on_char`. In --chat mode we wrap the prompt
    in the instruction template and stop at the EOT marker; otherwise it's a plain continuation.
    GPT.generate reuses its KV cache internally, so this is one forward step per character.
    """
    text = PROMPT_TEMPLATE.format(instruction=user_prompt) if args.chat else user_prompt
    idx = mx.array(encode(text) or encode("\n"))[None]
    if args.chat:
        streamer = StopStreamer(EOT, on_char)
        model.generate(idx, args.tokens, on_token=lambda t: streamer.feed(decode([t])), **gen_kwargs)
        streamer.close()
    else:
        model.generate(idx, args.tokens, on_token=lambda t: on_char(decode([t])) or False, **gen_kwargs)


def run_once(model, encode, decode, args, gen_kwargs):
    print(args.prompt, end="\n" if args.chat else "", flush=True)   # echo the prompt/instruction
    generate_reply(model, encode, decode, args.prompt, args, gen_kwargs,
                   on_char=lambda ch: print(ch, end="", flush=True))
    print()


def run_interactive(model, encode, decode, args, gen_kwargs):
    mode = "chat (instructions)" if args.chat else "continuation"
    print(f"interactive {mode} mode — type and press Enter; Ctrl-D or Ctrl-C to quit.")
    print(f"(temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, "
          f"rep_penalty={args.repetition_penalty}, {args.tokens} chars per reply)")
    while True:
        try:
            prompt = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not args.chat:
            print(prompt, end="", flush=True)             # echo the seed, then stream the rest
        generate_reply(model, encode, decode, prompt, args, gen_kwargs,
                       on_char=lambda ch: print(ch, end="", flush=True))
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
    p.add_argument("--entropy", action="store_true",
                   help="entropy-based ('entropix') sampling: adapt to the model's own "
                        "uncertainty instead of fixed temperature (ignores --temperature/--top_k/"
                        "--top_p). Tune via the ent_*/vent_* knobs in config.py.")
    p.add_argument("-i", "--interactive", action="store_true",
                   help="load the model once and keep prompting in a loop, streaming output")
    p.add_argument("--chat", action="store_true",
                   help="treat the prompt as an INSTRUCTION (wrap it in the SFT template and "
                        "stop at the end marker). Use with an SFT checkpoint from sft.py.")
    p.add_argument("--quantize", type=int, default=0, choices=[0, 2, 3, 4, 6, 8],
                   help="quantize weights to this many bits for smaller/faster inference "
                        "(0 = off; try 4 or 8)")
    p.add_argument("--q_group_size", type=int, default=64,
                   help="quantization group size (columns sharing one scale); must divide the "
                        "weight widths")
    p.add_argument("--ckpt", type=str, default="ckpt.npz")
    p.add_argument("--meta", type=str, default="ckpt.json")
    args = p.parse_args()

    model, encode, decode = load_model(args.ckpt, args.meta,
                                       quantize=args.quantize, q_group_size=args.q_group_size)
    # Collect the sampling controls once; 0/off values become None so generate() skips them.
    gen_kwargs = dict(
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        top_p=args.top_p if args.top_p > 0 else None,
        repetition_penalty=args.repetition_penalty,
        entropy_sampling=args.entropy,
    )

    if args.interactive:
        run_interactive(model, encode, decode, args, gen_kwargs)
    else:
        run_once(model, encode, decode, args, gen_kwargs)


if __name__ == "__main__":
    main()
