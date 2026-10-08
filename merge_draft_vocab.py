#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["tokenizers==0.22.2"]
# ///
"""Merge one or more draft-vocabulary counts files into a draft vocabulary list.

Run tools/draft_vocab.py with --counts-out in each place that has text (this only needs the
tokenizer and the corpus, no final --size decisions), collect the JSON files, then merge:

  python tools/merge_draft_vocab.py TOKENIZER OUT.txt --size 32768 --min-count 10 \
      'remote-site-A/counts.json' 'remote-site-B/counts.json' ...

TOKENIZER is a local tokenizer.json path or a Hugging Face repo id (`org/model` or
`org/model@revision`); the latter is fetched via the `hf` CLI into
$TF_TOKENIZER_CACHE (default /tmp/tokenizers)/<repo>/tokenizer.json.

Selection is exactly tools/draft_vocab.py's: all ids below --keep-below, optionally the tokenizer's
added tokens, then the merged corpus counts in frequency order down to --min-count, and the lowest
unused ids fill a short list. Merging is plain count addition, so the result equals what a single
run over the union of the corpora would produce (stats are summed per file).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import subprocess
import sys

from tokenizers import Tokenizer


def _cache_root() -> str:
    return os.environ.get("TF_TOKENIZER_CACHE") or "/tmp/tokenizers"


def resolve_tokenizer(arg: str) -> str:
    """Return a local tokenizer.json path for `arg`.

    `arg` is either an existing file path or an HF repo id (`org/model` or
    `org/model@revision`). Fetches via the `hf` CLI on first use.
    """
    if os.path.isfile(arg):
        return arg
    repo_id, sep, revision = arg.partition("@")
    repo_id = repo_id.strip()
    if not repo_id or sep and not revision:
        sys.exit(f"tf-util: tokenizer {arg!r} is neither an existing file nor a "
                 f"valid Hugging Face repo id (expected `org/model` or `org/model@revision`)")
    target_dir = os.path.join(_cache_root(), repo_id)
    target = os.path.join(target_dir, "tokenizer.json")
    if os.path.isfile(target):
        return target
    hf = shutil.which("hf")
    if hf is None:
        sys.exit(f"tf-util: tokenizer {arg!r} is a Hugging Face repo id but the `hf` "
                 f"CLI is not installed (curl -LsSf https://hf.co/cli/install.sh | bash -)")
    os.makedirs(target_dir, exist_ok=True)
    cmd = [hf, "download", repo_id, "tokenizer.json", "--local-dir", target_dir, "--quiet"]
    if revision:
        cmd += ["--revision", revision]
    subprocess.run(cmd, check=True)
    if not os.path.isfile(target):
        sys.exit(f"tf-util: `hf download` reported success but {target!r} is missing")
    return target


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tokenizer", help="path to tokenizer.json OR a Hugging Face repo id "
                   "(`org/model` or `org/model@revision`)")
    p.add_argument("counts", nargs="+", help="counts JSON files written by draft_vocab.py")
    p.add_argument("--out", default="draft_vocab.txt", metavar="FILE",
                   help="output vocab txt (default: draft_vocab.txt)")
    p.add_argument("--size", type=int, default=32768)
    p.add_argument("--keep-below", type=int, default=1024, help="always keep ids below this (special tokens)")
    p.add_argument("--min-count", type=int, default=10,
                   help="leave out ids seen fewer times (rare names and one-off words); the id prefix fills the rest")
    p.add_argument("--added-tokens", action="store_true", help="always keep the tokenizer's added (special) tokens")
    args = p.parse_args()

    tok_path = resolve_tokenizer(args.tokenizer)
    tok = Tokenizer.from_file(tok_path)
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
        keep |= {t["id"] for t in json.loads(open(tok_path).read())["added_tokens"]}
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
    print(json.dumps({"wrote": args.out, "counts_files": len(args.counts), "files": files,
                      "tokens": tokens, "size": len(ids),
                      "coverage": round(covered / max(1, tokens), 4),
                      "from_counts": sum(1 for t, c in counts.items() if c >= args.min_count and t in keep)}))


if __name__ == "__main__":
    main()
