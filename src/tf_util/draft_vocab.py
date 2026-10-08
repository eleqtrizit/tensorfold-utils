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

    def update(self, path: str, files: int, tokens: int, note: str = "") -> None:
        if not sys.stderr.isatty():
            return
        cols = shutil.get_terminal_size((80, 24)).columns
        spin = _SPINNER[self._i % len(_SPINNER)]
        self._i += 1
        head = f"{spin} {files} files · {tokens:,} tokens · "
        body = _truncate(path, max(8, cols - len(head) - 1))
        if note:  # e.g. '— skipped (binary)' shown on the same line
            body = _truncate(path, max(8, cols - len(head) - len(note) - 2))
            body += f" {note}"
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
    # Caches / logs / temp
    ".cache/", "coverage/", ".sass-cache", ".terraform/", ".serverless/",
    ".DS_Store", "Thumbs.db", ".log", ".tmp", ".bak",
    ".env", ".pem", "id_rsa",
)

# Non-prose file extensions, checked as an O(1) set lookup on the filename's
# extension (lowercased) BEFORE any substring matching — extension hits short-
# circuit the whole exclusion path. Deliberately NOT substrings: `.json` as a
# substring would wrongly match `index.json.js`, and `.o` would kill `todo.org`.
# Composites are spelled out: `.jsonl`, `.ndjson`, `.docx`, `.tar.gz` (-> .gz).
EXCLUDE_EXTENSIONS = frozenset({
    # tabular
    ".csv", ".tsv", ".psv", ".parquet", ".avro", ".orc", ".feather", ".arrow",
    ".xlsx", ".xls", ".ods",
    # config / structured
    ".json", ".jsonl", ".ndjson", ".yaml", ".yml", ".toml", ".xml", ".ini",
    ".cfg", ".conf",
    # databases
    ".db", ".sqlite", ".sqlite3", ".mdb", ".sql",
    # ML / serialized
    ".h5", ".hdf5", ".pkl", ".pickle", ".npy", ".npz", ".pt", ".pth", ".ckpt",
    ".safetensors", ".onnx", ".gguf", ".bin", ".msgpack", ".bson", ".cbor",
    ".pb", ".prototxt", ".caffemodel", ".mlmodel", ".mlpackage",
    # archives / packages / disk images
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".tbz2", ".xz", ".txz", ".zst",
    ".7z", ".rar", ".cab", ".lz", ".lz4", ".br",
    ".iso", ".img", ".dmg", ".vhd", ".vhdx", ".qcow2", ".vmdk",
    ".deb", ".rpm", ".msi", ".apk", ".ipa", ".appimage", ".snap", ".flatpak",
    ".crx", ".whl", ".egg", ".gem", ".nupkg",
    # images (raster / raw / design; .svg kept — it's text)
    ".png", ".jpg", ".jpeg", ".jpe", ".jfif", ".gif", ".bmp", ".dib", ".ico",
    ".cur", ".webp", ".tiff", ".tif", ".psd", ".psb", ".xcf", ".ai", ".indd",
    ".heic", ".heif", ".avif", ".svgz", ".cr2", ".cr3", ".nef", ".arw", ".dng",
    ".raf", ".orf", ".rw2", ".exr", ".hdr", ".pic", ".tga",
    # audio
    ".mp3", ".mp2", ".m4a", ".m4b", ".aac", ".opus", ".wav", ".wave", ".flac",
    ".ogg", ".oga", ".ogx", ".opus", ".wma", ".aiff", ".aif", ".au", ".ape",
    ".wv", ".mka", ".mid", ".midi", ".caf", ".dsf", ".amr", ".ac3", ".dts",
    # video
    ".mp4", ".m4v", ".mpg", ".mpeg", ".webm", ".avi", ".mkv", ".mov", ".wmv",
    ".flv", ".mts", ".m2ts", ".3gp", ".3g2", ".ogv", ".vob", ".rm",
    ".rmvb", ".asf", ".divx", ".f4v",
    # documents
    ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xlsb", ".epub", ".mobi",
    ".azw", ".azw3", ".djvu", ".xps", ".odt", ".rtf",
    # fonts / compiled / native code (note: .ts intentionally absent — TypeScript is prose)
    ".woff", ".woff2", ".ttf", ".otf", ".eot", ".so", ".dylib", ".dll", ".exe",
    ".sys", ".drv", ".ocx", ".ax", ".lib", ".pdb", ".idb", ".elf",
    ".ko", ".o", ".a", ".obj", ".wasm", ".pyc", ".pyo", ".class", ".jar",
    ".war", ".ear", ".node", ".rlib", ".crate",
    # 3D / CAD / design assets
    ".stl", ".fbx", ".glb", ".blend", ".3ds", ".dae", ".usd", ".usda",
    ".usdc", ".usdz", ".skp", ".dwg", ".dxf",
})


def _ext_excluded(path: str) -> bool:
    """O(1): lowercase the path, look up its extension in EXCLUDE_EXTENSIONS."""
    _, ext = os.path.splitext(path.lower())
    return ext in EXCLUDE_EXTENSIONS


def expand_paths(paths: list[str], excludes: list[str] | None = None) -> list[str]:
    """Expand each path: a file is itself, a directory is walked recursively, a glob is expanded.

    Exclusion order: the O(1) extension check (`EXCLUDE_EXTENSIONS`) runs first on
    every candidate, then the substring patterns (`ALWAYS_EXCLUDE` + user `-e`).
    """
    excludes = excludes or []

    def excluded(path: str) -> bool:
        return _ext_excluded(path) or any(pat in path for pat in ALWAYS_EXCLUDE) \
            or any(pat in path for pat in excludes)

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
    """Cheap binary sniff (git's heuristic): read the first `sniff` bytes and
    reject on a NUL byte or known magic numbers. Binary blobs almost always
    contain a NUL early; real text virtually never does.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(sniff)
    except OSError:
        return False
    if not head:
        return True                        # empty file: nothing to tokenize anyway
    if b"\x00" in head:
        return False
    # common magic numbers: images, archives, executables, pdf, sqlite, wasm, etc.
    magics = (b"\x89PNG", b"\xff\xd8\xff", b"GIF8", b"BM", b"\x00\x00\x01\x00",
              b"\x00\x00\x02\x00",          # png, jpeg, gif, bmp, ico, cur
              b"PK\x03\x04", b"\x1f\x8b", b"\x04\x22\x4d\x18",   # zip, gzip, lz4
              b"7z\xbc\xaf\x27\x1c", b"Rar!", b"\xfd7zXZ\x00",  # 7z, rar, xz
              b"\x7fELF", b"MZ", b"\xfe\xed\xfa",          # elf, pe, mach-o
              b"%PDF", b"SQLite format 3\x00", b"\x00asm")   # pdf, sqlite, wasm
    if any(head.startswith(m) for m in magics):
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tf-util make-vocab")
    p.add_argument("tokenizer", help="path to tokenizer.json OR a Hugging Face repo id "
                   "(`org/model` or `org/model@revision`); fetched into $TF_TOKENIZER_CACHE "
                   f"(default {cache_root()}/<repo>/tokenizer.json) on first use")
    p.add_argument("-p", "--path", action="append", required=True, metavar="PATH",
                   help="corpus file, directory (walked recursively), or glob (repeatable)")
    p.add_argument("-e", "--exclude", action="append", default=[], metavar="STRING",
                   help="skip files whose path contains STRING (repeatable, e.g. -e .venv -e node_modules)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="print each skipped file (binary, non-utf-8, unreadable, empty) to stderr")
    p.add_argument("-o", "--out", default=None, metavar="FILE",
                   help="output counts JSON (default: <hostname>.json)")
    args = p.parse_args(argv)

    tok_path = resolve_tokenizer(args.tokenizer)
    tok = Tokenizer.from_file(tok_path)
    counts: collections.Counter = collections.Counter()
    total = used = skipped = 0
    status = Status()
    for path in expand_paths(args.path, args.exclude):
        if not looks_like_text(path):
            skipped += 1
            status.update(path, used, total, note="skipped (binary)")
            if args.verbose:
                print(f"tf-util: skipped (binary): {path}", file=sys.stderr)
            continue
        status.update(path, used, total)
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        except UnicodeDecodeError:
            skipped += 1
            if args.verbose:
                print(f"tf-util: skipped (not utf-8): {path}", file=sys.stderr)
            continue
        except (OSError, IsADirectoryError):
            skipped += 1
            if args.verbose:
                print(f"tf-util: skipped (unreadable): {path}", file=sys.stderr)
            continue
        if not text:
            if args.verbose:
                print(f"tf-util: skipped (empty): {path}", file=sys.stderr)
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
    print(json.dumps({"wrote": out, "files": used, "tokens": total, "skipped": skipped}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
