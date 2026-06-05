"""
vision.py — the "eyes" for the GPT: turning images into the model's own vector language.

THIS FILE IS THE MULTIMODAL DEMO. Read it after model.py. The single idea it exists to teach:

    A transformer does not process *text*. It processes a sequence of VECTORS (lists of n_embd
    numbers). In model.py the only thing that turns text into those vectors is one line —
    `self.token_emb(idx)`. Everything after it (attention, the blocks, the LM head) just shuffles
    vectors around and has no idea they came from characters.

    So to make the model "see", we build a SECOND converter that turns an image into the same kind
    of vectors, and splice them into the same sequence. The transformer then attends over image
    and text together, identically. That converter is `PatchEmbed` below, and the splice happens
    via the `prefix=` hook we added to GPT.__call__.

How an image becomes vectors (the `PatchEmbed`):
    An image is too big to be one vector, so we chop it into small squares ("patches") — a 24x24
    image with 8x8 patches becomes a 3x3 grid = 9 patches. Each patch's pixels are flattened and
    pushed through ONE Linear layer down to n_embd numbers. Patches are to an image what tokens are
    to text: the unit we turn into a vector. (This is exactly how a Vision Transformer works, and
    it is why people say "an image is worth N tokens".)

The toy task: we draw simple colored shapes (a red circle, a green square, a blue triangle, ...)
and ask the model to CAPTION them. The whole point is that it is tiny, needs no download, and you
can print the pixels and the patches yourself. Captions are predicted one character at a time by
the very same GPT.

Run `python vision.py` to see a few generated images (rendered in your terminal with color) next
to their captions, then `python train_mm.py` to train, then `python sample_mm.py` to watch it
caption fresh images.
"""
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from config import Config
from data import CharTokenizer
from model import GPT

# --- the toy "world": three shapes x three colors = nine things to describe -----------------
SHAPES = ["circle", "square", "triangle"]
COLORS = {"red": (1.0, 0.0, 0.0), "green": (0.0, 1.0, 0.0), "blue": (0.0, 0.0, 1.0)}
COLOR_NAMES = list(COLORS)

# The text scaffold. We give the model a fixed lead-in ("Caption: ") after the image, then it must
# produce "a <color> <shape>." and stop at the end marker. EOT is the same "<|endoftext|>" string
# the rest of the repo uses as a natural stop signal, so it feels familiar.
CAPTION_PREFIX = "Caption: "
EOT = "<|endoftext|>"


def caption_for(shape: str, color: str) -> str:
    """The ground-truth caption for a (shape, color), e.g. 'a red circle'."""
    return f"a {color} {shape}"


def caption_corpus() -> str:
    """
    Every character the captioning task will ever need, concatenated, so a CharTokenizer built on
    it has the full vocabulary. (The char tokenizer's vocab is just the sorted unique characters.)
    """
    parts = [CAPTION_PREFIX, EOT]
    for color in COLOR_NAMES:
        for shape in SHAPES:
            parts.append(caption_for(shape, color) + ".")
    return "\n".join(parts)


def build_caption_tokenizer() -> CharTokenizer:
    """The same char tokenizer as the text model, but trained on the captions' tiny alphabet."""
    return CharTokenizer(caption_corpus())


# --- drawing the images ---------------------------------------------------------------------

def make_image(shape: str, color: str, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """
    Paint one shape of one color onto a black RGB image, with a randomized size and position so
    the model can't just memorize a single fixed picture. Returns a (H, W, 3) float32 array in
    [0, 1]. Pure numpy and pure geometry — open it up and print it; there is no magic here.
    """
    H = W = cfg.img_size
    img = np.zeros((H, W, cfg.img_channels), dtype=np.float32)
    rgb = COLORS[color]

    r = int(rng.integers(H // 5, H // 3))      # the shape's radius / half-width, in pixels
    cy = int(rng.integers(r, H - r))           # center, kept far enough from the edges to fit
    cx = int(rng.integers(r, W - r))
    yy, xx = np.mgrid[0:H, 0:W]                 # per-pixel row/col coordinates

    if shape == "circle":
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    elif shape == "square":
        mask = (np.abs(yy - cy) <= r) & (np.abs(xx - cx) <= r)
    else:                                       # triangle, apex pointing up
        ty = yy - (cy - r)                      # 0 at the apex row, 2r at the base
        half_width = np.where((ty >= 0) & (ty <= 2 * r), ty / (2 * r) * r, -1.0)
        mask = np.abs(xx - cx) <= half_width    # widens linearly from apex to base
    img[mask] = rgb
    return img


def ascii_image(image: np.ndarray, row_step: int = 2) -> str:
    """
    Render an image in the terminal using 24-bit color background blocks, so you can actually SEE
    what the model is captioning. Two spaces per pixel (terminal chars are ~twice as tall as wide,
    so this keeps shapes looking square); we skip every other row to keep the picture compact.
    """
    H, W, _ = image.shape
    lines = []
    for y in range(0, H, row_step):
        cells = []
        for x in range(0, W):
            r, g, b = (np.clip(image[y, x], 0, 1) * 255).astype(int)
            cells.append(f"\x1b[48;2;{r};{g};{b}m  \x1b[0m")
        lines.append("".join(cells))
    return "\n".join(lines)


# --- turning an image into model vectors: the patch embedding -------------------------------

def patchify(images: mx.array, patch: int) -> mx.array:
    """
    Chop a batch of images into flattened patches.
    (B, H, W, C) -> (B, n_patches, patch*patch*C), where n_patches = (H/patch)*(W/patch).
    The reshape+transpose just regroups the pixels so each patch's values land contiguously.
    """
    B, H, W, C = images.shape
    nph, npw = H // patch, W // patch                     # patches down, patches across
    x = images.reshape(B, nph, patch, npw, patch, C)      # split H and W into (n, patch) each
    x = x.transpose(0, 1, 3, 2, 4, 5)                     # (B, nph, npw, patch, patch, C)
    return x.reshape(B, nph * npw, patch * patch * C)     # one flat vector per patch


class PatchEmbed(nn.Module):
    """
    The image analogue of `token_emb`. token_emb maps a token id -> an n_embd vector via a lookup
    table; PatchEmbed maps a patch of pixels -> an n_embd vector via a single Linear. After this,
    a patch and a character are indistinguishable to the rest of the model: both are just C-dim
    vectors in the sequence.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.img_size % cfg.patch_size == 0, "img_size must be divisible by patch_size"
        self.patch = cfg.patch_size
        self.n_patches = (cfg.img_size // cfg.patch_size) ** 2
        patch_dim = cfg.patch_size * cfg.patch_size * cfg.img_channels   # pixels in one patch
        self.proj = nn.Linear(patch_dim, cfg.n_embd)      # the only learned weights here

    def __call__(self, images: mx.array) -> mx.array:
        patches = patchify(images, self.patch)            # (B, n_patches, patch_dim)
        return self.proj(patches)                         # (B, n_patches, C) — same C as text!


# --- the captioning model: PatchEmbed + the unmodified GPT ----------------------------------

class CaptionModel(nn.Module):
    """
    The whole multimodal model: a PatchEmbed "eye" and an ordinary GPT, side by side. The GPT is
    the SAME class as the text-only model — we change nothing inside it. All the multimodal-ness is
    "embed the image, hand the vectors to the GPT as a prefix":

        image -> PatchEmbed -> prefix vectors ┐
                                              ├─> GPT([prefix ; text]) -> next-char logits
        "Caption: " text -> token_emb --------┘

    Training grades only the caption characters (loss masking, just like sft.py). At generation we
    prefill with the image + "Caption: " once, then sample one character at a time with the KV
    cache, exactly like text generation.
    """

    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()
        self.patch_embed = PatchEmbed(cfg)
        self.gpt = GPT(cfg, vocab_size)

    def __call__(self, images: mx.array, idx: mx.array) -> mx.array:
        """Training forward: returns logits over the full (image patches + text) sequence."""
        prefix = self.patch_embed(images)                 # (B, n_patches, C)
        logits, _, _ = self.gpt(idx, prefix=prefix)       # (B, n_patches + T_text, vocab)
        return logits

    def caption(self, image: np.ndarray, encode, decode, max_new: int = 40,
                temperature: float = 0.0) -> str:
        """
        Generate a caption for one (H, W, 3) image. We embed the image, seed the text with the
        "Caption: " lead-in, and let the GPT continue character by character until it emits the EOT
        marker. temperature 0 = greedy (deterministic) — right for a task with one correct answer.
        """
        prefix = self.patch_embed(mx.array(image[None]))  # (1, n_patches, C)
        ids = mx.array(encode(CAPTION_PREFIX))[None]       # (1, T_prompt)

        # First pass: process the image + the prompt together and build the KV cache.
        logits, _, caches = self.gpt(ids, prefix=prefix)
        text = ""
        for _ in range(max_new):
            last = logits[:, -1, :]                        # scores for the next character
            if temperature == 0.0:
                nxt = int(mx.argmax(last, axis=-1)[0])
            else:
                nxt = int(mx.random.categorical(last / temperature)[0])
            text += decode([nxt])
            if EOT in text:                                # the model said "I'm done"
                return text.split(EOT)[0].strip()
            # Feed just the new character back in; the cache means this is one cheap step.
            logits, _, caches = self.gpt(mx.array([[nxt]]), caches=caches)
        return text.strip()


# --- batching for training ------------------------------------------------------------------

def encode_caption(encode, shape: str, color: str, caption_len: int):
    """
    Tokenize one example into (ids, mask), each length caption_len+1 (so a left-shift gives the
    next-token targets). mask is 1 on the CAPTION characters (incl. EOT) and 0 on the "Caption: "
    lead-in — so, just like SFT, the model is graded only on what it should generate.
    """
    prompt_ids = encode(CAPTION_PREFIX)
    resp_ids = encode(caption_for(shape, color) + "." + EOT)
    ids = prompt_ids + resp_ids
    mask = [0] * len(prompt_ids) + [1] * len(resp_ids)
    ids, mask = ids[: caption_len + 1], mask[: caption_len + 1]
    ids += [0] * (caption_len + 1 - len(ids))             # right-pad to a fixed length
    mask += [0] * (caption_len + 1 - len(mask))
    return ids, mask


def make_batch(batch_size: int, encode, cfg: Config, rng: np.random.Generator):
    """
    Draw `batch_size` random (image, caption) examples and pack them for next-token training.
    Returns (images, x, y, m): images (B,H,W,3); x/y the shifted caption ids (B, caption_len);
    m the loss weight (B, caption_len), 1 only on caption tokens.
    """
    images, xs, ys, ms = [], [], [], []
    for _ in range(batch_size):
        shape = SHAPES[int(rng.integers(len(SHAPES)))]
        color = COLOR_NAMES[int(rng.integers(len(COLOR_NAMES)))]
        images.append(make_image(shape, color, cfg, rng))
        ids, mask = encode_caption(encode, shape, color, cfg.caption_len)
        xs.append(ids[:-1])                                # inputs
        ys.append(ids[1:])                                 # the next char at each position
        ms.append(mask[1:])                                # loss weight for predicting each y
    images = mx.array(np.stack(images))                    # (B, H, W, 3)
    x = mx.array(xs)                                       # (B, caption_len) int
    y = mx.array(ys)
    m = mx.array(np.array(ms, dtype=np.float32))           # (B, caption_len) float
    return images, x, y, m


if __name__ == "__main__":
    # A quick look at the dataset: render a few images with their captions, and show what the
    # patch grid looks like. No model involved — this is just the raw inputs.
    cfg = Config()
    rng = np.random.default_rng(cfg.seed)
    tok = build_caption_tokenizer()
    print(f"caption vocab ({tok.vocab_size} chars): {''.join(tok.itos[i] for i in range(tok.vocab_size))!r}")
    pe = PatchEmbed(cfg)
    print(f"image {cfg.img_size}x{cfg.img_size} -> {pe.n_patches} patches "
          f"-> {pe.n_patches} vectors of size {cfg.n_embd} (the GPT's width)\n")
    for _ in range(3):
        shape = SHAPES[int(rng.integers(len(SHAPES)))]
        color = COLOR_NAMES[int(rng.integers(len(COLOR_NAMES)))]
        img = make_image(shape, color, cfg, rng)
        print(ascii_image(img))
        print(f"  caption: {caption_for(shape, color)}\n")
