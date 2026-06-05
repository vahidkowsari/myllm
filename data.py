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
import json
import os
import urllib.request

import mlx.core as mx
import numpy as np

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


def _token_cache_paths():
    """Where the tokenized corpus is cached on disk (a binary blob + a small JSON sidecar)."""
    base = config.data_path.rsplit(".", 1)[0]
    return f"{base}.{config.tokenizer}.tokens.bin", f"{base}.{config.tokenizer}.tokens.json"


def load_tokens(text: str, tokenizer):
    """
    Tokenize the whole corpus ONCE and cache the integer ids to disk, then hand back a
    memory-mapped view of them. Memory-mapping means the ids live on disk and the OS pages in
    only the slices we actually touch — so a corpus far bigger than RAM still works, and
    re-running training doesn't re-encode the whole thing. (On the small default corpus this is
    just a convenience; it's the kind of thing that matters once you scale the data up.)

    The cache is keyed by tokenizer + vocab size + corpus length; change any of those and it is
    rebuilt automatically. ids are stored as uint16 when the vocab fits (it does for char/BPE
    here), which halves the file vs. int32.
    """
    bin_path, meta_path = _token_cache_paths()
    key = {"tokenizer": config.tokenizer, "vocab_size": tokenizer.vocab_size, "chars": len(text)}

    if os.path.exists(bin_path) and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if all(meta.get(k) == v for k, v in key.items()):
            return np.memmap(bin_path, dtype=meta["dtype"], mode="r")   # reuse the cache

    print(f"tokenizing corpus -> {bin_path} (cached for next time)")
    dtype = "uint16" if tokenizer.vocab_size <= 65536 else "int32"
    ids = np.asarray(tokenizer.encode(text), dtype=dtype)
    ids.tofile(bin_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({**key, "dtype": dtype, "n_tokens": int(ids.size)}, f)
    return np.memmap(bin_path, dtype=dtype, mode="r")


def load_data():
    """
    Returns (train_data, val_data, tokenizer).
    train/val are 1-D, disk-backed (memory-mapped) arrays of token ids — the entire corpus as
    integers, split 90/10. Training grabs random windows out of these (see get_batch).
    """
    text = download_if_needed()
    tokenizer = build_tokenizer(text)
    ids = load_tokens(text, tokenizer)        # np.memmap, 1-D — stays on disk until indexed

    n = int(0.9 * len(ids))
    train_data, val_data = ids[:n], ids[n:]   # views into the memmap; still disk-backed
    return train_data, val_data, tokenizer


def get_batch(data, block_size: int, batch_size: int):
    """
    Pull a random batch of (input, target) pairs.

    For a chunk of text c[0..block_size], the model's job at every position t is to predict
    c[t+1] from c[0..t]. So:
        x = data[i      : i+block_size]      # inputs
        y = data[i+1    : i+block_size+1]    # the same sequence shifted left by one
    Stacking `batch_size` random windows gives tensors of shape (batch_size, block_size).

    We pick `batch_size` random start positions with MLX's RNG (so config.seed controls it),
    then gather that (batch_size, block_size) grid of windows from the disk-backed array and move
    just those small batches onto the GPU as int32 MLX arrays. Only the windows we sample are
    ever read from disk — that's the point of the memmap.
    """
    ix = np.asarray(mx.random.randint(0, len(data) - block_size, shape=(batch_size,)))
    rows = ix[:, None] + np.arange(block_size)[None, :]   # (batch_size, block_size) of indices
    x = mx.array(np.asarray(data[rows], dtype=np.int32))
    y = mx.array(np.asarray(data[rows + 1], dtype=np.int32))
    return x, y


if __name__ == "__main__":
    train_data, val_data, tok = load_data()
    print(f"tokenizer        : {config.tokenizer}")
    print(f"vocab size       : {tok.vocab_size} tokens")
    print(f"train tokens     : {len(train_data):,}")
    print(f"val tokens       : {len(val_data):,}")
    print(f"first 20 tokens  : {tok.decode(train_data[:20].tolist())!r}")
    print(f"as token ids     : {train_data[:20].tolist()}")
