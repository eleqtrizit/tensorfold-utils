"""Count corpus token frequencies for a draft vocabulary.

Counting only — all selection limits live in merge_draft_vocab.py.
Writes the raw per-id counts as a portable, mergeable JSON file named after
the host by default (e.g. vyper.json).
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import shutil
import socket
import sys
import time

from tokenizers import Tokenizer

from tf_util.hf_tokenizer import resolve_tokenizer, cache_root

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _truncate(path: str, width: int) -> str:
    """Shorten `path` to fit `width` columns, leading with … if cut."""
    if len(path) <= width:
        return path
    return "…" + path[-(width - 1):]


class Status:
    """Single-line live status on stderr that overwrites itself with \\r."""

    def __init__(self) -> None:
        self._i = 0
        self._t0 = time.perf_counter()

    def update(self, path: str, files: int, tokens: int) -> None:
        if not sys.stderr.isatty():
            return
        cols = shutil.get_terminal_size((80, 24)).columns
        spin = _SPINNER[self._i % len(_SPINNER)]
        self._i += 1
        head = f"{spin} {files} files · {tokens:,} tokens · "
        body = _truncate(path, max(8, cols - len(head) - 1))
        line = head + body
        # pad to clear any leftover chars from a longer previous line
        sys.stderr.write("\r" + line.ljust(cols - 1)[: cols - 1])
        sys.stderr.flush()

    def clear(self) -> None:
        if not sys.stderr.isatty():
            return
        cols = shutil.get_terminal_size((80, 24)).columns
        sys.stderr.write("\r" + " " * (cols - 1) + "\r")
        sys.stderr.flush()


def expand_paths(paths: list[str], excludes: list[str] | None = None) -> list[str]:
    """Expand each path: a file is itself, a directory is walked recursively, a glob is expanded.

    `excludes` are substrings: any path containing one is dropped (e.g. -e .venv -e node_modules).
    """
    excludes = excludes or []

    def excluded(path: str) -> bool:
        return any(pat in path for pat in excludes)

    files: set[str] = set()
    for path in paths:
        if os.path.isfile(path):
            if not excluded(path):
                files.add(path)
        elif os.path.isdir(path):
            files.update(os.path.join(root, name)
                         for root, _, names in os.walk(path) for name in names
                         if not excluded(os.path.join(root, name)))
        else:
            matched = glob.glob(path, recursive=True)
            files.update(f for f in matched if os.path.isfile(f) and not excluded(f))
    return sorted(files)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tf-util make-vocab")
    p.add_argument("tokenizer", help="path to tokenizer.json OR a Hugging Face repo id "
                   "(`org/model` or `org/model@revision`); fetched into $TF_TOKENIZER_CACHE "
                   f"(default {cache_root()}/<repo>/tokenizer.json) on first use")
    p.add_argument("-p", "--path", action="append", required=True, metavar="PATH",
                   help="corpus file, directory (walked recursively), or glob (repeatable)")
    p.add_argument("-e", "--exclude", action="append", default=[], metavar="STRING",
                   help="skip files whose path contains STRING (repeatable, e.g. -e .venv -e node_modules)")
    p.add_argument("-o", "--out", default=None, metavar="FILE",
                   help="output counts JSON (default: <hostname>.json)")
    args = p.parse_args(argv)

    tok_path = resolve_tokenizer(args.tokenizer)
    tok = Tokenizer.from_file(tok_path)
    counts: collections.Counter = collections.Counter()
    total = used = 0
    status = Status()
    for path in expand_paths(args.path, args.exclude):
        status.update(path, used, total)
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
    status.clear()

    host = socket.gethostname()
    out = args.out or f"{host}.json"
    with open(out, "w") as handle:
        json.dump({"host": host,
                   "count_files": used, "count_tokens": total,
                   "counts": {str(k): v for k, v in sorted(counts.items())}}, handle)
    print(json.dumps({"wrote": out, "files": used, "tokens": total}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
