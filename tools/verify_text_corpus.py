#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify a text corpus against the real text-only distillation loader, and report its size.

    python tools/verify_text_corpus.py                                  # auto-locate the corpus
    python tools/verify_text_corpus.py --corpus /workspace/ppx_text_corpus
    python tools/verify_text_corpus.py --tokenizer /path/tokenizer_spm_32k_3.model   # exact token count

Every check here goes through `distill.data.text_dataset` itself -- the module `distill/train.py` imports --
rather than re-implementing the format, so a pass means the trainer will read this corpus.

Token count: with `--tokenizer`, this is the EXACT count the trainer will see (it calls the same
`build_token_cache`). Without it, a public 32,000-piece SentencePiece model (google-t5/t5-small's
`spiece.model`, the same vocabulary size as PersonaPlex's `tokenizer_spm_32k_3.model`) is used as a reference
and the number is reported as an estimate. The teacher tokenizer is a gated download, so the exact count is
only available on the pod.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "moshi"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np                                                    # noqa: E402
from distill.data.text_dataset import (                               # noqa: E402
    TEXT_CARD, SEPARATOR, TEXT_SUFFIXES, _iter_documents, build_token_cache, TextTokenStream,
)


def find_corpus() -> str:
    for c in ("/workspace/ppx_text_corpus", str(REPO / "ppx_text_corpus")):
        if Path(c).is_dir():
            return c
    raise SystemExit("No corpus found. Pass --corpus, or run tools/build_text_corpus.py first.")


def reference_tokenizer() -> tuple[str, str]:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("google-t5/t5-small", "spiece.model")
    return path, "reference (google-t5/t5-small spiece.model, 32,000 pieces)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--tokenizer", default=None, help="teacher tokenizer_spm_32k_3.model for an exact count")
    ap.add_argument("--seq-len", type=int, default=512, help="must match --text-seq-len (default 512)")
    args = ap.parse_args()

    corpus = args.corpus or find_corpus()
    print(f"corpus: {corpus}\n")

    # ---- 1. files the loader will discover ---------------------------------------------------------------
    files = sorted(f for f in Path(corpus).rglob("*") if f.suffix.lower() in TEXT_SUFFIXES)
    print("1. FILES THE LOADER DISCOVERS")
    if not files:
        raise SystemExit(f"   none -- the loader needs {'/'.join(TEXT_SUFFIXES)} files under {corpus}")
    for f in files:
        print(f"   {f.stat().st_size / 1e6:>8.2f} MB  {f.relative_to(corpus)}")
    non_shard = [f for f in files if f.suffix.lower() != ".jsonl"]
    if non_shard:
        print(f"   NOTE: {len(non_shard)} discovered file(s) are not .jsonl shards; step 2 shows whether "
              f"they contribute documents: {[f.name for f in non_shard]}")

    # ---- 2. documents, through _iter_documents ------------------------------------------------------------
    docs = list(_iter_documents([corpus]))
    chars = sum(len(d) for d in docs)
    words = sum(len(d.split()) for d in docs)
    print(f"\n2. DOCUMENTS (via distill.data.text_dataset._iter_documents)")
    print(f"   documents : {len(docs):,}")
    print(f"   characters: {chars:,}")
    print(f"   words     : {words:,}")
    lens = sorted(len(d) for d in docs)
    if lens:
        q = lambda p: lens[min(len(lens) - 1, int(p * len(lens)))]      # noqa: E731
        print(f"   doc chars : min {lens[0]}  p25 {q(.25)}  median {q(.5)}  p75 {q(.75)}  max {lens[-1]}")
    empty = sum(1 for d in docs if not d.strip())
    print(f"   empty docs: {empty} (skipped by build_token_cache)")

    # ---- 3. tokenize with the SAME function the trainer calls --------------------------------------------
    if args.tokenizer:
        tok, kind = args.tokenizer, "EXACT (teacher tokenizer_spm_32k_3.model)"
    else:
        tok, kind = reference_tokenizer()
    exact = bool(args.tokenizer)
    print(f"\n3. TOKENIZATION -- {kind}")
    td = tempfile.mkdtemp(prefix="ppx_verify_tok_")
    try:
        cache = str(Path(td) / "verify_tokens.npy")
        arr = build_token_cache([corpus], tok, cache)
        n_tok = int(arr.shape[0])
        id3 = int((np.asarray(arr) == SEPARATOR).sum())
        print(f"   tokens        : {n_tok:,}" + ("" if exact else "   (ESTIMATE -- see the note below)"))
        print(f"   dtype         : {arr.dtype} (uint16 required; text_card={TEXT_CARD} fits)")
        print(f"   max token id  : {int(arr.max()):,} (must be < {TEXT_CARD})")
        if exact:
            print(f"   separators    : {id3:,} occurrences of id {SEPARATOR} "
                  f"(>= {len(docs):,}, one appended per document)")
        else:
            # With a reference tokenizer, id 3 is an ordinary piece of ITS vocabulary, not PersonaPlex's
            # padding id, so counting it says nothing about document boundaries.
            print(f"   separators    : not meaningful with a reference tokenizer "
                  f"(id {SEPARATOR} is a normal piece there); {len(docs):,} will be appended with the "
                  f"teacher tokenizer")
        print(f"   chars / token : {chars / max(n_tok, 1):.2f}")
        assert arr.dtype == np.uint16, arr.dtype
        assert int(arr.max()) < TEXT_CARD

        # ---- 4. the stream the trainer actually iterates -------------------------------------------------
        print(f"\n4. TextTokenStream(seq_len={args.seq_len})")
        stream = TextTokenStream(arr, args.seq_len, seed=1234)
        b1 = stream.batch(step=7, rank=0, world=1, micro=2, accum=4)
        b2 = stream.batch(step=7, rank=0, world=1, micro=2, accum=4)
        b3 = stream.batch(step=8, rank=0, world=1, micro=2, accum=4)
        print(f"   windows       : {len(stream):,}")
        print(f"   batch shape   : {b1.shape}  dtype {b1.dtype}")
        assert b1.shape == (8, args.seq_len + 1) and b1.dtype == np.int64
        assert np.array_equal(b1, b2), "not resume-deterministic"
        assert not np.array_equal(b1, b3), "different steps gave the same batch"
        r0 = stream.batch(step=0, rank=0, world=2, micro=1, accum=2)
        r1 = stream.batch(step=0, rank=1, world=2, micro=1, accum=2)
        assert not np.array_equal(r0, r1), "ranks not disjoint"
        print("   determinism   : OK (same step -> same batch; ranks disjoint)")
        del arr, stream                      # release the memmap before removing the temp dir (Windows)
    finally:
        import shutil
        shutil.rmtree(td, ignore_errors=True)

    # ---- 5. topic coverage, from the provenance fields ---------------------------------------------------
    man = Path(corpus) / "corpus_manifest.json"
    if man.exists():
        m = json.loads(man.read_text(encoding="utf-8"))
        print("\n5. TOPIC COVERAGE (from corpus_manifest.json)")
        for topic, s in m.get("by_topic", {}).items():
            print(f"   {topic:24s} {s['documents']:>5,} docs  {s['characters']:>9,} chars")

    print("\n" + "=" * 78)
    print("COMPATIBLE: the corpus is read by distill/data/text_dataset.py as-is.")
    print(f"Run with:  --text-data {corpus} --tokenizer <teacher tokenizer_spm_32k_3.model> "
          f"--text-seq-len {args.seq_len}")
    if not args.tokenizer:
        print("\nNOTE: the token count above is a reference-tokenizer ESTIMATE. The exact count is printed by "
              "the notebook's Section 7b cell (and this script with --tokenizer) on the pod, where the gated "
              "teacher tokenizer is available.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
