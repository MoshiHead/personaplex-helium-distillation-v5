# SPDX-License-Identifier: MIT
"""Plain-text token stream for TEXT-ONLY distillation (`distill/train.py --text-data ...`).

Why this exists
---------------
The student's temporal transformer is the only part of PersonaPlex that holds world knowledge (the frozen
depformer / `linears` / `text_linear` / Mimi are the acoustic rendering stack), and the distillation init
reduces it to a fraction of the teacher's width, depth, heads and FFN channels. Recovering the text behaviour
it lost needs text tokens -- a lot of them.

An audio conversation is a terrible carrier for those tokens. At 12.5 Hz, a sample of 1000 frames costs 1000
transformer positions *plus* a 16-codebook depformer forward at every position, and carries only ~100 real
text tokens (the other ~90% of the text stream is PAD/EPAD and the forced persona prompt). A 100 h corpus of
such conversations contains well under 1M agent text tokens in total.

`moshi.models.lm.ScaledEmbedding` makes a much cheaper path available: it returns EXACTLY zero for
`zero_idx == LMModel.zero_token_id == -1`. So filling the 16 audio rows of a `[B, 17, T]` code tensor with -1
turns `embed_codes` into a pure text embedding lookup, and `forward_codes` into a plain text-LM forward
through the same `transformer` -> `bridge` -> frozen `out_norm` -> frozen `text_linear` chain -- one position
per text token, no Mimi, no depformer, no 17-stream delay machinery.

This module turns text files into the flat token array that mode consumes.

What text to use
----------------
Because the objective is KL against the teacher, the corpus only has to COVER the domain -- it does not need
answers in it. The teacher supplies the targets. So: general prose for broad competence, plus text about
whatever the student must be able to discuss (for this project: cryptocurrency / blockchain / finance, and
whatever identity facts the personas assert).

Layout on disk
--------------
`build_token_cache` concatenates every input document into one `uint16` .npy array (`text_card = 32000` fits
in uint16; so does `text_initial_token_id = 32000`), with `SEPARATOR` between documents. `TextTokenStream`
then serves fixed-length windows from it, addressed by global step so a resumed run reads the same data in
the same order -- the same contract `GlobalBatchSampler` provides for the audio split.
"""

from pathlib import Path
import json
import logging
import typing as tp

import numpy as np

logger = logging.getLogger(__name__)

# `existing_text_padding_id = 3` (moshi/models/lm.py) doubles as the between-documents separator: it is the
# token the model already reads as "nothing being said", so it is a natural document break and it never
# collides with a real SentencePiece piece.
SEPARATOR = 3
TEXT_CARD = 32000

TEXT_SUFFIXES = (".txt", ".md", ".jsonl", ".json")


def _iter_documents(paths: tp.Sequence[str]) -> tp.Iterator[str]:
    """Yield document strings from files/directories/globs.

    `.txt` / `.md` -> one document per file. `.jsonl` -> one document per line, read from the first present
    key among `text`, `content`, `answer`, `output`, `completion` (so Dolly/OASST-style dumps work unchanged);
    a line that is a bare JSON string is taken as-is. `.json` -> a list of such objects or strings.
    """
    files: list[Path] = []
    for p in paths:
        path = Path(p)
        if any(ch in p for ch in "*?["):
            files += sorted(Path().glob(p))
        elif path.is_dir():
            files += sorted(f for f in path.rglob("*") if f.suffix.lower() in TEXT_SUFFIXES)
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"--text-data entry matches nothing: {p}")
    if not files:
        raise FileNotFoundError(f"--text-data matched no {'/'.join(TEXT_SUFFIXES)} files: {list(paths)}")

    def from_obj(obj) -> tp.Optional[str]:
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            for k in ("text", "content", "answer", "output", "completion", "response"):
                if isinstance(obj.get(k), str):
                    return obj[k]
        return None

    for f in files:
        suffix = f.suffix.lower()
        if suffix in (".txt", ".md"):
            yield f.read_text(encoding="utf-8", errors="replace")
        elif suffix == ".jsonl":
            with open(f, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = from_obj(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                    if doc:
                        yield doc
        else:  # .json
            try:
                data = json.loads(f.read_text(encoding="utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            for obj in (data if isinstance(data, list) else [data]):
                doc = from_obj(obj)
                if doc:
                    yield doc


def build_token_cache(
    text_paths: tp.Sequence[str],
    tokenizer_model: str,
    cache_path: str,
    max_tokens: tp.Optional[int] = None,
) -> np.ndarray:
    """Tokenize `text_paths` with the PersonaPlex SentencePiece model into one flat uint16 .npy array.

    Cached: an existing `cache_path` is reused as-is (tokenizing a large corpus is slow, and every rank would
    otherwise redo it). Delete the file to force a rebuild.
    """
    cache = Path(cache_path)
    if cache.exists():
        arr = np.load(cache, mmap_mode="r")
        logger.info("text cache: reusing %s (%d tokens)", cache, arr.shape[0])
        return arr

    import sentencepiece

    sp = sentencepiece.SentencePieceProcessor(tokenizer_model)
    chunks: list[np.ndarray] = []
    total = 0
    n_docs = 0
    for doc in _iter_documents(text_paths):
        doc = doc.strip()
        if not doc:
            continue
        ids = sp.encode(doc)
        if not ids:
            continue
        assert max(ids) < TEXT_CARD, f"token id {max(ids)} >= text_card {TEXT_CARD} -- wrong tokenizer?"
        chunks.append(np.asarray(ids + [SEPARATOR], dtype=np.uint16))
        total += len(ids) + 1
        n_docs += 1
        if max_tokens is not None and total >= max_tokens:
            break
    if not chunks:
        raise ValueError(f"No text was tokenized from {list(text_paths)}")
    arr = np.concatenate(chunks)
    if max_tokens is not None:
        arr = arr[:max_tokens]
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    tmp.replace(cache)
    logger.info("text cache: wrote %s (%d documents, %d tokens)", cache, n_docs, arr.shape[0])
    return np.load(cache, mmap_mode="r")


class TextTokenStream:
    """Fixed-length text windows addressed by global step, so resuming reproduces the data order exactly.

    `batch(step, rank, world, micro, accum)` returns the windows this rank must process at `step`: an
    `int64` array of shape `[micro * accum, seq_len + 1]`. The extra +1 column is the next-token target for
    the last position, so a caller can use `w[:, :-1]` as input and `w[:, 1:]` as targets without losing a
    position at the window boundary.
    """

    def __init__(self, tokens: np.ndarray, seq_len: int, seed: int = 1234):
        self.tokens = tokens
        self.seq_len = int(seq_len)
        self.seed = int(seed)
        self.n_windows = max(1, (tokens.shape[0] - 1) // self.seq_len)
        if tokens.shape[0] < self.seq_len + 1:
            raise ValueError(
                f"text corpus has {tokens.shape[0]} tokens, needs at least --text-seq-len + 1 = "
                f"{self.seq_len + 1}. Point --text-data at more text.")
        self._perm_cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return self.n_windows

    def _perm(self, epoch: int) -> np.ndarray:
        if epoch not in self._perm_cache:
            if len(self._perm_cache) > 3:
                self._perm_cache.pop(min(self._perm_cache))
            self._perm_cache[epoch] = np.random.default_rng([self.seed, 777, epoch]).permutation(self.n_windows)
        return self._perm_cache[epoch]

    def _window(self, pos: int) -> np.ndarray:
        epoch, off = divmod(pos, self.n_windows)
        w = int(self._perm(epoch)[off])
        start = w * self.seq_len
        return np.asarray(self.tokens[start:start + self.seq_len + 1], dtype=np.int64)

    def batch(self, step: int, rank: int, world: int, micro: int, accum: int) -> np.ndarray:
        per_step = world * micro * accum
        base = step * per_step + rank * micro * accum
        return np.stack([self._window(base + i) for i in range(micro * accum)])
