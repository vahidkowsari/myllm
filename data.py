"""
data.py — the *tokenizer* and data loader, the very lowest layer.

An LLM never sees text. It sees integers. This file is where text becomes integers and back.

By default we use a CHARACTER-LEVEL tokenizer: one token = one character. That means there is
zero mystery in the vocabulary — it is literally the sorted set of unique characters in the
training file. Set `tokenizer = "bpe"` in config.py to instead train a small subword tokenizer
(see bpe.py): the same idea applied to chunks of characters, so one token covers several of
them and the context window stretches much further.

Run this file directly once to download the corpus and print tokenizer stats:

    python data.py
"""
import os
import urllib.request

import mlx.core as mx

from config import config


def download_if_needed(path: str = config.data_path, url: str = config.data_url) -> str:
    """Fetch the raw training text the first time, then read it from disk on later runs."""
    if not os.path.exists(path):
        print(f"downloading corpus -> {path}")
        urllib.request.urlretrieve(url, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class CharTokenizer:
    """
    The whole tokenizer. It builds two lookup tables from the unique characters:
      stoi: string/char -> integer id
      itos: integer id  -> string/char
    `encode` maps a string to a list of ids; `decode` maps ids back to a string.
    """

    def __init__(self, text: str):
        chars = sorted(set(text))          # the vocabulary, e.g. ['\n', ' ', '!', ...]
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str) -> list[int]:
        # Silently drop characters not in the vocab (e.g. a stray emoji at sampling time) so a
        # single odd keystroke can't crash encoding. Training text only contains known chars.
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def to_meta(self) -> dict:
        # The vocabulary as one string, in id order: char at index i has token id i.
        return {"type": "char",
                "vocab": "".join(self.itos[i] for i in range(self.vocab_size))}

    @classmethod
    def from_meta(cls, meta: dict):
        self = cls.__new__(cls)                 # skip __init__ — we restore from saved vocab
        vocab = meta["vocab"]
        self.vocab_size = len(vocab)
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for i, ch in enumerate(vocab)}
        return self


def build_tokenizer(text: str):
    """Construct the tokenizer chosen in config (and train it on `text`)."""
    if config.tokenizer == "char":
        return CharTokenizer(text)
    if config.tokenizer == "bpe":
        from bpe import BPETokenizer                       # imported lazily; char path needs nothing
        return BPETokenizer().train(text, config.bpe_vocab_size)
    raise ValueError(f"unknown tokenizer {config.tokenizer!r} (use 'char' or 'bpe')")


def load_tokenizer(meta: dict):
    """Rebuild a saved tokenizer from its to_meta() dict (used by sample.py)."""
    if meta["type"] == "char":
        return CharTokenizer.from_meta(meta)
    if meta["type"] == "bpe":
        from bpe import BPETokenizer
        return BPETokenizer.from_meta(meta)
    raise ValueError(f"unknown tokenizer type {meta['type']!r}")


def load_data():
    """
    Returns (train_data, val_data, tokenizer).
    train/val are 1-D int32 MLX arrays of token ids — the entire corpus as integers, split
    90/10. Training will grab random windows out of these.
    """
    text = download_if_needed()
    tokenizer = build_tokenizer(text)
    ids = mx.array(tokenizer.encode(text))

    n = int(0.9 * len(ids))
    train_data, val_data = ids[:n], ids[n:]
    return train_data, val_data, tokenizer


def get_batch(data: mx.array, block_size: int, batch_size: int):
    """
    Pull a random batch of (input, target) pairs.

    For a chunk of text c[0..block_size], the model's job at every position t is to predict
    c[t+1] from c[0..t]. So:
        x = data[i      : i+block_size]      # inputs
        y = data[i+1    : i+block_size+1]    # the same sequence shifted left by one
    Stacking `batch_size` random windows gives tensors of shape (batch_size, block_size).

    We pick `batch_size` random start positions, then build a (batch_size, block_size) grid of
    indices and gather all the windows at once — no Python loop needed.
    """
    ix = mx.random.randint(0, len(data) - block_size, shape=(batch_size,))
    rows = ix[:, None] + mx.arange(block_size)[None, :]   # (batch_size, block_size) of indices
    x = data[rows]
    y = data[rows + 1]
    return x, y


if __name__ == "__main__":
    train_data, val_data, tok = load_data()
    print(f"tokenizer        : {config.tokenizer}")
    print(f"vocab size       : {tok.vocab_size} tokens")
    print(f"train tokens     : {len(train_data):,}")
    print(f"val tokens       : {len(val_data):,}")
    print(f"first 20 tokens  : {tok.decode(train_data[:20].tolist())!r}")
    print(f"as token ids     : {train_data[:20].tolist()}")
