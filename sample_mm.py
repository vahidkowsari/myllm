"""
sample_mm.py — caption fresh images with a trained multimodal model.

Loads the CaptionModel saved by train_mm.py, generates random shape images, renders each in the
terminal (in color), and prints the model's caption next to the truth. This is the multimodal
analogue of sample.py: feed an input the model has never seen and watch it respond — except the
input is a picture, not a text prompt.

    python sample_mm.py            # caption 5 random images
    python sample_mm.py --n 10     # ...10 of them
    python sample_mm.py --seed 7   # different random images
"""
import argparse
import json

import numpy as np

from config import Config
from data import load_tokenizer
from vision import (CaptionModel, make_image, ascii_image, caption_for, SHAPES, COLOR_NAMES)


def load_caption_model(ckpt_path: str, meta_path: str):
    """Rebuild the tokenizer + config from the JSON sidecar, then load the weights once."""
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    tokenizer = load_tokenizer(meta["tokenizer"])
    cfg = Config(**meta["config"])
    model = CaptionModel(cfg, vocab_size=tokenizer.vocab_size)
    model.load_weights(ckpt_path)
    model.eval()
    return model, cfg, tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5, help="how many random images to caption")
    p.add_argument("--seed", type=int, default=2024, help="seed for the random images")
    p.add_argument("--ckpt", type=str, default="ckpt_mm.npz")
    p.add_argument("--meta", type=str, default="ckpt_mm.json")
    args = p.parse_args()

    model, cfg, tokenizer = load_caption_model(args.ckpt, args.meta)
    rng = np.random.default_rng(args.seed)

    correct = 0
    for _ in range(args.n):
        shape = SHAPES[int(rng.integers(len(SHAPES)))]
        color = COLOR_NAMES[int(rng.integers(len(COLOR_NAMES)))]
        img = make_image(shape, color, cfg, rng)
        pred = model.caption(img, tokenizer.encode, tokenizer.decode)
        truth = caption_for(shape, color)
        ok = pred.rstrip(".") == truth
        correct += ok
        print(ascii_image(img))
        print(f"  pred: {pred}    {'✓' if ok else '✗ (true: ' + truth + ')'}\n")
    print(f"{correct}/{args.n} correct")


if __name__ == "__main__":
    main()
