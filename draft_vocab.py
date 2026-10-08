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
import time

from tokenizers import Tokenizer

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _truncate(path: str, width: int) -> str:
    if len(path) <= width:
        return path
    return "…" + path[-(width - 1):]


class Status:
    """Single-line live status on stderr that overwrites itself with \\r."""

    def __init__(self) -> None:
        self._i = 0

    def update(self, path: str, files: int, tokens: int) -> None:
        if not sys.stderr.isatty():
            return
        cols = shutil.get_terminal_size((80, 24)).columns
        spin = _SPINNER[self._i % len(_SPINNER)]
        self._i += 1
        head = f"{spin} {files} files · {tokens:,} tokens · "
        body = _truncate(path, max(8, cols - len(head) - 1))
        line = head + body
        sys.stderr.write("\r" + line.ljust(cols - 1)[: cols - 1])
        sys.stderr.flush()

    def clear(self) -> None:
        if not sys.stderr.isatty():
            return
        cols = shutil.get_terminal_size((80, 24)).columns
        sys.stderr.write("\r" + " " * (cols - 1) + "\r")
        sys.stderr.flush()


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


# Always skipped, in the background, on top of any -e patterns: VCS internals and
# dependency/build caches — machine-generated, language-skewed, huge, and pure noise for
# corpus stats. Substring match, so e.g. `.git` catches `.git/hooks/pre-commit` too.
ALWAYS_EXCLUDE = (
    # VCS internals
    ".git", ".hg", ".svn",
    # JS ecosystems
    "node_modules", "bower_components/", ".next/", ".nuxt/", ".yarn/", ".pnp.js",
    "npm-debug.log", "yarn-error.log",
    # Python
    ".venv", "venv/", ".tox", ".nox", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".hypothesis",
    "site-packages/", ".eggs/", "egg-info", ".ipynb_checkpoints",
    "pip-log.txt", "poetry.lock",
    # Build artifacts
    "target/", "build/", "dist/", "out/", ".gradle/", "cmake-build-debug",
    # Vendored / IDE / editor
    "vendor/", "_vendor/", ".idea/", ".vscode/", ".swp", ".swo", ".swn",
    # Agent / AI tool state and caches
    ".claude", ".pi", ".agents", ".codex", ".aider*", ".cursor", ".gemini",
    ".copilot*", ".continue", ".windsurf", ".codeium", ".tabnine",
    # Caches / logs / temp / secrets
    ".cache/", "coverage/", ".sass-cache", ".terraform/", ".serverless/",
    ".DS_Store", "Thumbs.db", ".log", ".tmp", ".bak",
    ".env", ".pem", "id_rsa",
)


def expand_paths(paths: list[str], excludes: list[str] | None = None) -> list[str]:
    """Expand paths; `excludes` are substrings dropped from the result (e.g. -e .venv).
    ALWAYS_EXCLUDE is applied on top of them unconditionally.
    """
    excludes = list(ALWAYS_EXCLUDE) + (excludes or [])

    def excluded(path: str) -> bool:
        return any(pat in path for pat in excludes)

    files: set[str] = set()
    for path in paths:
        if os.path.isfile(path):
            if not excluded(path):
                files.add(path)
        elif os.path.isdir(path):
            for root, dirs, names in os.walk(path):
                # prune excluded dirs in place so os.walk never descends into them
                dirs[:] = [d for d in dirs if not excluded(os.path.join(root, d, ''))]
                files.update(f for f in (os.path.join(root, name) for name in names)
                             if not excluded(f))
        else:
            matched = glob.glob(path, recursive=True)
            files.update(f for f in matched if os.path.isfile(f) and not excluded(f))
    return sorted(files)


def looks_like_text(path: str, sniff: int = 8192) -> bool:
    """Cheap binary sniff (git's heuristic): NUL byte or known magic numbers in the
    first `sniff` bytes -> binary. Binary blobs almost always contain a NUL early.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(sniff)
    except OSError:
        return False
    if not head:
        return True
    if b"\x00" in head:
        return False
    magics = (b"\x89PNG", b"\xff\xd8\xff", b"GIF8", b"BM", b"\x00\x00\x01\x00",
              b"\x00\x00\x02\x00",
              b"PK\x03\x04", b"\x1f\x8b", b"\x04\x22\x4d\x18",
              b"7z\xbc\xaf\x27\x1c", b"Rar!", b"\xfd7zXZ\x00",
              b"\x7fELF", b"MZ", b"\xfe\xed\xfa",
              b"%PDF", b"SQLite format 3\x00", b"\x00asm")
    if any(head.startswith(m) for m in magics):
        return False
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("tokenizer", help="path to tokenizer.json OR a Hugging Face repo id "
                   "(`org/model` or `org/model@revision`)")
    p.add_argument("-p", "--path", action="append", required=True, metavar="PATH",
                   help="corpus file, directory (walked recursively), or glob (repeatable)")
    p.add_argument("-e", "--exclude", action="append", default=[], metavar="STRING",
                   help="skip files whose path contains STRING (repeatable, e.g. -e .venv -e node_modules)")
    p.add_argument("-o", "--out", default=None, metavar="FILE",
                   help="output counts JSON (default: <hostname>.json)")
    args = p.parse_args()

    tok_path = resolve_tokenizer(args.tokenizer)
    tok = Tokenizer.from_file(tok_path)
    counts: collections.Counter = collections.Counter()
    total = used = 0
    status = Status()
    for path in expand_paths(args.path, args.exclude):
        status.update(path, used, total)
        if not looks_like_text(path):
            continue
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
