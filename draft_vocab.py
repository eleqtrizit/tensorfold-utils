#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["tokenizers==0.22.2"]
# ///
"""Count corpus token frequencies for a draft vocabulary (portable standalone copy).

Counting only — all selection limits live in merge_draft_vocab.py. Writes a
portable, mergeable counts JSON named after the host by default (vyper.json):

  uv run draft_vocab.py TOKENIZER -p 'corpus/**/*.py' [-p more/] [-o vyper.json]

  TOKENIZER may be a path to tokenizer.json or a Hugging Face repo id
  (`org/model` or `org/model@revision`); the latter is fetched via the `hf` CLI
  into $TF_TOKENIZER_CACHE (default /tmp/tokenizers)/<repo>/tokenizer.json.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import shutil
import socket
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
    p.add_argument("tokenizer", help="path to tokenizer.json OR a Hugging Face repo id "
                   "(`org/model` or `org/model@revision`)")
    p.add_argument("-p", "--path", action="append", required=True, metavar="PATH",
                   help="corpus file, directory (walked recursively), or glob (repeatable)")
    p.add_argument("-o", "--out", default=None, metavar="FILE",
                   help="output counts JSON (default: <hostname>.json)")
    args = p.parse_args()

    tok_path = resolve_tokenizer(args.tokenizer)
    tok = Tokenizer.from_file(tok_path)
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
    print(json.dumps({"wrote": out, "files": used, "tokens": total}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
