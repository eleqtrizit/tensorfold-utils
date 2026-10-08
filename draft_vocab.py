#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["tokenizers==0.22.2"]
# ///
"""Count corpus token frequencies for a draft vocabulary (portable standalone copy).

Counting only — all selection limits live in merge_draft_vocab.py. Writes a
portable, mergeable counts JSON named after the host by default (vyper.json):

  uv run draft_vocab.py TOKENIZER_JSON -p 'corpus/**/*.py' [-p more/] [-o vyper.json]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import socket
import sys

from tokenizers import Tokenizer


def expand_paths(paths: list[str]) -> list[str]:
    files: set[str] = set()
    for path in paths:
        if os.path.isfile(path):
            files.add(path)
        elif os.path.isdir(path):
            files.update(os.path.join(root, name)
                         for root, _, names in os.walk(path) for name in names)
        else:
            matched = glob.glob(path, recursive=True)
            files.update(f for f in matched if os.path.isfile(f))
    return sorted(files)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("tokenizer")
    p.add_argument("-p", "--path", action="append", required=True, metavar="PATH",
                   help="corpus file, directory (walked recursively), or glob (repeatable)")
    p.add_argument("-o", "--out", default=None, metavar="FILE",
                   help="output counts JSON (default: <hostname>.json)")
    args = p.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    counts: collections.Counter = collections.Counter()
    total = used = 0
    for path in expand_paths(args.path):
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        except (UnicodeDecodeError, OSError, IsADirectoryError):
            continue
        if not text:
            continue
        ids = tok.encode(text).ids
        counts.update(ids)
        total += len(ids)
        used += 1

    host = socket.gethostname()
    out = args.out or f"{host}.json"
    with open(out, "w") as handle:
        json.dump({"host": host,
                   "count_files": used, "count_tokens": total,
                   "counts": {str(k): v for k, v in sorted(counts.items())}}, handle)
    print(json.dumps({"out": out, "files": used, "tokens": total}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
