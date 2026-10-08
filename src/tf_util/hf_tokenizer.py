"""Resolve a tokenizer argument to a local `tokenizer.json` path.

The token-consuming commands (`make-vocab`, `merge`) take a tokenizer as their
first positional argument. That argument may now be either:

  * a path to an existing `tokenizer.json` on disk, or
  * a Hugging Face repo id (e.g. `nvidia/Nemotron-Super-3.5-GA-FINAL-row105-QAD-PreStitched-BoostedMTP`),
    optionally pinned to a revision with the `repo@revision` syntax
    (e.g. `org/model@main` or `org/model@<commit-sha>`).

For a repo id we download just `tokenizer.json` with the `hf` CLI into a cache
root (default `/tmp/tokenizers`, override with `TF_TOKENIZER_CACHE`) under
`<cache>/<repo_id>/tokenizer.json`, skipping the download when the file already
exists. The tokenizer revision must be identical on every machine — pin it with
`@revision` and hash the file if reproducibility matters.

We shell out to the `hf` CLI rather than depending on `huggingface_hub` so the
standalone PEP 723 scripts (which only pin `tokenizers`) keep working on
machines that have the `hf` CLI but not the python package.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

DEFAULT_CACHE = "/tmp/tokenizers"


def cache_root() -> str:
    return os.environ.get("TF_TOKENIZER_CACHE") or DEFAULT_CACHE


def resolve_tokenizer(arg: str) -> str:
    """Return a local `tokenizer.json` path for `arg`.

    `arg` is either an existing file path or an HF repo id (`org/model` or
    `org/model@revision`). Downloads via `hf` only when the cached file is
    missing. Exits the process with code 1 on any failure.
    """
    if os.path.isfile(arg):
        return arg

    repo_id, sep, revision = arg.partition("@")
    repo_id = repo_id.strip()
    if not repo_id or sep and not revision:
        # Empty id, or `repo@` with no revision.
        _die(f"tokenizer {arg!r} is neither an existing file nor a valid "
             f"Hugging Face repo id (expected `org/model` or `org/model@revision`)")

    target_dir = os.path.join(cache_root(), repo_id)
    target = os.path.join(target_dir, "tokenizer.json")
    if os.path.isfile(target):
        return target

    hf = shutil.which("hf")
    if hf is None:
        _die(f"tokenizer {arg!r} is a Hugging Face repo id but the `hf` CLI is "
             f"not installed; install it (curl -LsSf https://hf.co/cli/install.sh | bash -) "
             f"or pass a local tokenizer.json path")

    os.makedirs(target_dir, exist_ok=True)
    cmd = [hf, "download", repo_id, "tokenizer.json",
           "--local-dir", target_dir, "--quiet"]
    if revision:
        cmd += ["--revision", revision]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        _die(f"`hf download` failed for {arg!r} (exit {exc.returncode})")

    if not os.path.isfile(target):
        _die(f"`hf download` reported success but {target!r} is missing; "
             f"the repo may not contain a `tokenizer.json` at its root "
             f"(some models ship it as e.g. `tokenizer/tokenizer.json`)")
    return target


def _die(msg: str) -> None:
    print(f"tf-util: {msg}", file=sys.stderr)
    sys.exit(1)
