"""Merge draft-vocabulary counts files and select a draft vocabulary.

Standalone logic for merge_draft_vocab.py (portable copy of TensorFold's
tools/merge_draft_vocab.py).
"""

from __future__ import annotations

import argparse
import collections
import json
import sys

from tokenizers import Tokenizer


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tf-util merge")
    p.add_argument("tokenizer")
    p.add_argument("out")
    p.add_argument("counts", nargs="+", help="counts JSON files written by make-vocab --counts-out")
    p.add_argument("--size", type=int, default=32768)
    p.add_argument("--keep-below", type=int, default=1024, help="always keep ids below this (special tokens)")
    p.add_argument("--min-count", type=int, default=10,
                   help="leave out ids seen fewer times (rare names and one-off words); the id prefix fills the rest")
    p.add_argument("--added-tokens", action="store_true", help="always keep the tokenizer's added (special) tokens")
    args = p.parse_args(argv)

    tok = Tokenizer.from_file(args.tokenizer)
    counts: collections.Counter = collections.Counter()
    files = tokens = 0
    for path in args.counts:
        with open(path, encoding="utf-8") as handle:
            blob = json.load(handle)
        counts.update({int(k): v for k, v in blob["counts"].items()})
        files += blob.get("count_files", 0)
        tokens += blob.get("count_tokens", 0)

    keep = set(range(args.keep_below))
    if args.added_tokens:
        keep |= {t["id"] for t in json.loads(open(args.tokenizer).read())["added_tokens"]}
    for tid, count in counts.most_common():
        if len(keep) >= args.size or count < args.min_count:
            break
        keep.add(tid)
    fill = iter(range(args.keep_below, tok.get_vocab_size()))
    while len(keep) < args.size:                       # the id prefix fills a short list
        keep.add(next(fill))
    ids = sorted(keep)
    covered = sum(c for t, c in counts.items() if t in keep)
    with open(args.out, "w") as handle:
        handle.write("\n".join(str(i) for i in ids) + "\n")
    print(json.dumps({"counts_files": len(args.counts), "files": files, "tokens": tokens, "size": len(ids),
                      "coverage": round(covered / max(1, tokens), 4),
                      "from_counts": sum(1 for t, c in counts.items() if c >= args.min_count and t in keep)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
