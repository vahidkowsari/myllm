"""
bpe.py — a tiny, readable Byte-Pair Encoding tokenizer (in the spirit of Karpathy's minbpe).

The char tokenizer in data.py uses one token per character: maximally transparent, but it
means the model spends its whole context window on individual letters. Real LLMs use *subword*
tokenizers, and BPE is the classic one. The idea is dead simple:

    1. Start with the raw bytes of the text (256 possible byte values = the base vocab).
    2. Find the most frequent adjacent pair of tokens and MERGE it into a single new token.
    3. Repeat until you reach the target vocab size.

So common chunks like "the", " and", "ing" become single tokens, and one token now covers
several characters — the same block_size suddenly sees much more text.

We work at the BYTE level (vocab starts at the 256 byte values) so *any* text encodes without
an "unknown token", and decoding is just gluing bytes back together. This is exactly how GPT-2's
tokenizer works underneath, minus the regex pre-splitting and the special tokens.
"""


def get_stats(ids):
    """Count how often each adjacent pair of token ids occurs. {(a, b): count}."""
    counts = {}
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def merge(ids, pair, new_id):
    """Return a copy of `ids` with every occurrence of `pair` replaced by the single `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """
    Byte-level BPE. After `train`, it holds:
      merges: {(a, b): new_id}  the learned merges, IN THE ORDER they were learned (this order
              is what `encode` replays — earliest merges take priority).
      vocab:  {id: bytes}       what each token id expands to, for decoding.
    """

    def __init__(self):
        self.merges = {}                                   # (a, b) -> new_id
        self.vocab = {i: bytes([i]) for i in range(256)}   # the 256 base byte tokens

    @property
    def vocab_size(self):
        return len(self.vocab)

    def train(self, text: str, vocab_size: int):
        """Learn merges from `text` until the vocab reaches `vocab_size` (>= 256)."""
        assert vocab_size >= 256, "byte-level BPE needs at least the 256 base byte tokens"
        ids = list(text.encode("utf-8"))
        for i in range(vocab_size - 256):
            stats = get_stats(ids)
            if not stats:
                break                                      # text fully merged into one token
            pair = max(stats, key=stats.get)              # the most frequent adjacent pair
            new_id = 256 + i
            ids = merge(ids, pair, new_id)                 # collapse it everywhere
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        return self

    def encode(self, text: str) -> list[int]:
        """Text -> token ids. Repeatedly apply the EARLIEST-learned merge that still matches."""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            # Among the pairs currently present, pick the one we learned first (lowest new_id).
            # `inf` for pairs we never merged makes them lose the min().
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break                                      # nothing left to merge
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids) -> str:
        """Token ids -> text. Glue each token's bytes together, then UTF-8 decode."""
        data = b"".join(self.vocab[int(i)] for i in ids)
        return data.decode("utf-8", errors="replace")     # replace any split multibyte char

    # --- serialization (so train.py can save it and sample.py can rebuild it) ----------------

    def to_meta(self) -> dict:
        # Just the merge list in order; the vocab is reconstructable from it. Each entry is
        # [a, b, new_id] meaning "tokens a,b merge into new_id".
        return {"type": "bpe",
                "merges": [[a, b, nid] for (a, b), nid in self.merges.items()]}

    @classmethod
    def from_meta(cls, meta: dict):
        tok = cls()
        for a, b, nid in meta["merges"]:                   # replay merges in saved order
            tok.merges[(a, b)] = nid
            tok.vocab[nid] = tok.vocab[a] + tok.vocab[b]
        return tok


if __name__ == "__main__":
    # Quick demo: train a small BPE on a snippet and show the compression vs. raw bytes.
    sample = "the cat sat on the mat. the cat ate the rat. " * 20
    tok = BPETokenizer().train(sample, vocab_size=300)
    ids = tok.encode("the cat sat on the mat.")
    print(f"vocab size : {tok.vocab_size}")
    print(f"text bytes : {len('the cat sat on the mat.'.encode())}")
    print(f"bpe tokens : {len(ids)}  -> {ids}")
    print(f"round-trip : {tok.decode(ids)!r}")
