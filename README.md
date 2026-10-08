# tensorfold-utils

[![Repo](https://img.shields.io/badge/github-eleqtrizit%2Ftensorfold--utils-blue)](https://github.com/eleqtrizit/tensorfold-utils)

Standalone helpers for building **draft vocabularies** for [TensorFold](https://github.com/ashhart/TensorFold)
and the related vLLM work ([PR #59740](https://github.com/vllm-project/vllm/pull/59740)).
They deliberately live outside the main repo: **count a corpus wherever the text is, then merge the counts anywhere.**

## Background

In speculative/MTP decoding the draft model proposes tokens and the target model verifies them with its
full vocabulary. The draft head therefore only needs to score a *subset* of the vocab: a token missing from
the list can't be proposed, which costs a little acceptance length — **never correctness** (output stays
byte-identical to undrafted decoding). A smaller list means proportionally fewer `lm_head` rows to read,
which speeds up decode on bandwidth-bound hardware (DGX Spark, Macs).

The list is built by ranking token frequencies over a public text corpus, always keeping low IDs and
special tokens.

## Requirements

- [uv](https://docs.astral.sh/uv/) — `tokenizers==0.22.2` pinned, Python ≥ 3.11.
- The [`hf` CLI](https://huggingface.co/docs/huggingface_hub/cli) — only needed when you pass a
  Hugging Face repo id instead of a local `tokenizer.json` path (see below).

## Install

```bash
uv tool install git+https://github.com/eleqtrizit/tensorfold-utils   # from this repo
uv tool install /path/to/tensorfold-utils                            # or from a local checkout
```

This installs a `tf-util` executable on your PATH. The repo also ships standalone PEP 723 scripts
(`draft_vocab.py`, `merge_draft_vocab.py`) that can be copied to a machine with just uv —
run them with `uv run draft_vocab.py ...`.

## Usage

Everything goes through the single entry point `tf-util`, which dispatches to the two underlying
commands.

```bash
# Count a corpus on this machine; writes counts JSON (default: <hostname>.json, e.g. vyper.json)
tf-util make-vocab TOKENIZER -p PATH [-p PATH ...] [-o OUT.json]

# Sum counts files, then select the vocab (all limits live here); writes draft_vocab.txt by default
tf-util merge TOKENIZER COUNTS.json [...] [--out OUT.txt] [--size 32768] \
        [--min-count 10] [--keep-below 1024] [--added-tokens]
```

`TOKENIZER` is either a path to a local `tokenizer.json` **or** a Hugging Face repo id
(`org/model`, `org/model@revision`, or just `gpt2`). For a repo id, `tf-util` downloads only
`tokenizer.json` via the `hf` CLI into `$TF_TOKENIZER_CACHE` (default `/tmp/tokenizers`)/`<repo_id>`/`tokenizer.json`
on first use and reuses the cached file afterwards — so you no longer have to copy the tokenizer
to every machine, just name the model. Pin a revision with `@revision` (branch/tag/commit) so every
machine uses identical vocab IDs.

### `make-vocab`

Alias: `uv run draft_vocab.py ...` (self-contained PEP 723 script; logic in `src/tf_util/draft_vocab.py`).

Tokenizes the corpus files given via `-p` (repeatable: files, directories, or globs) and writes the
raw per-ID counts
raw per-ID counts as a portable, mergeable JSON file named after the host by default (`vyper.json`),
including a `host` key. Files whose path contains any `-e`/`--exclude` substring are skipped; a
built-in `ALWAYS_EXCLUDE` list is also applied in the background, always, with excluded dirs pruned
from the walk entirely — VCS internals (`.git`, `.hg`, `.svn`), JS ecosystems (`node_modules`,
`.next/`, `.nuxt/`, `.yarn/`), Python (`.venv`, `venv/`, `.tox`, `__pycache__`, mypy/pytest/ruff
caches, `site-packages/`, `.ipynb_checkpoints`), build outputs (`target/`, `build/`, `dist/`,
`out/`, `.gradle/`), vendored/IDE (`vendor/`, `.idea/`, `.vscode/`, swap files), AI/agent tool
state (`.claude`, `.pi`, `.agents`, `.codex`, `.aider*`, `.cursor`, `.gemini`, `.copilot*`,
`.continue`, `.windsurf`, `.codeium`, `.tabnine`), caches/logs/temp/secrets (`.cache/`,
`coverage/`, `.terraform/`, `.DS_Store`, `Thumbs.db`, `.log`, `.tmp`, `.bak`, `.env`, `.pem`,
`id_rsa`). Data/config/serialized/media extensions (`.csv`, `.json`, `.jsonl`, `.yaml`, `.toml`,
`.parquet`, `.avro`, `.xlsx`, `.db`, `.sqlite`, `.sql`, `.h5`, `.pkl`, `.npy`, `.pt`, `.ckpt`,
`.safetensors`, `.onnx`, `.gguf`, `.bin`, `.msgpack`/`.bson`/`.cbor`, archives/packages/disk images
(`.zip`, `.tar`, `.gz`, `.zst`, `.7z`, `.iso`, `.dmg`, `.apk`, `.deb`, `.rpm`, `.msi`, `.whl`, ...),
images (`.png`...`.heic`, `.avif`, `.psd`, camera RAW), audio (`.mp3`, `.m4a`, `.flac`, `.opus`,
`.wma`, `.mid`, ...), video (`.mp4`, `.mkv`, `.mov`, `.webm`, ...), documents (`.pdf`, Office,
`.epub`, `.mobi`, `.djvu`, `.rtf`), fonts, compiled/native code (`.exe`, `.dll`, `.so`, `.o`, `.a`,
`.obj`, `.pyc`, `.jar`, `.wasm`, `.pdb`, ...), and 3D/CAD (`.stl`, `.fbx`, `.glb`, `.blend`,
`.dwg`, ...). These live in a separate
`EXCLUDE_EXTENSIONS` frozenset, checked first as an O(1) set lookup on the lowercased filename
extension — unlike substring matching, `todo.org` (org-mode notes) and `index.android.js` are
safe, and `index.json.js` isn't wrongly skipped. Composites are spelled out: `.jsonl`, `.ndjson`,
`.docx`. The full list lives in `ALWAYS_EXCLUDE` / `EXCLUDE_EXTENSIONS` in `draft_vocab.py`. A cheap binary sniff (git's
heuristic: NUL byte or known magic numbers — PNG/zip/ELF/PDF/SQLite/… — in the first 8KB) rejects
non-text files before they're decoded, so blobs never poison the counts. Every skipped file is
tallied and reported as `"skipped": N` in the JSON; run with `-v`/`--verbose` to print each skipped
path and its reason (binary / not utf-8 / unreadable / empty) to stderr. Rejected files also flash
briefly on the live status line with a trailing `skipped (binary)` marker instead of being shown
as if they were being tokenized.
**This is the counting half of the split workflow** — no size/limit decisions
happen here; those all happen at merge time. A live status line (braille spinner + running `files`/
`tokens` counts + the current path) is drawn on stderr while it works, then cleared for the final
JSON (auto-disabled when stderr isn't a TTY, so piping stays clean).

### `merge`

Alias: `uv run merge_draft_vocab.py ...`, logic in `src/tf_util/merge_draft_vocab.py`.

Sums the counts from one or more counts files and selects the draft vocabulary: all IDs below
`--keep-below` (default 1024), optionally all tokenizer added tokens (`--added-tokens`), then
frequency-ranked IDs down to `--min-count`, padded to `--size` with the lowest unused IDs, writing
the newline-separated ID list to `--out` (default `draft_vocab.txt`). The result is
**byte-identical** to counting the union corpus in a single run (verified with `cmp`). The counts
files are small (one entry per appearing ID, KBs–MBs), so they ship easily.

Both commands print a JSON one-liner to stdout naming every file they wrote (`"wrote"` key) plus
stats; `merge`'s stats include `coverage` — the fraction of corpus tokens covered by the selected
vocab, i.e. the expected acceptance ceiling *on that corpus*.

## Workflow: corpus split across machines

1. On each machine (installed tool or standalone script) just name the model — no tokenizer copy needed:
   ```bash
   tf-util make-vocab nvidia/Nemotron-Super-3.5-GA-FINAL-row105-QAD-PreStitched-BoostedMTP -p code/ -o local.json
   ```
   Pin a revision with `@revision` if you want byte-reproducible counts. The `hf` CLI must be on
   PATH (and `HF_TOKEN` set for gated repos).
2. Ship the `local.json` files back (any channel).
3. Merge once, deciding size/min-count a single time (writes `draft_vocab.txt` by default):
   ```bash
   tf-util merge nvidia/Nemotron-Super-3.5-GA-FINAL-row105-QAD-PreStitched-BoostedMTP \
       --size 32768 --min-count 10 *.json
   ```

The output is a plain newline-separated integer ID list (the runtime pads to a multiple of 64 itself), so
it slots into TensorFold's `draft_ids()` / a vLLM `speculative_config.draft_token_map` / SGLang
`--speculative-token-map` style config.

## Caveats

- No file size limit anymore: every file matched by `-p` is read in full.
- Corpus choice is the quality lever: a CPython-stdlib-built list is code/English-skewed and loses
  acceptance on other languages. Build from traffic that resembles yours, and hash your inputs
  (`sha256sum` tokenizer + corpus manifest) if reproducibility matters — this was the repo's own pitfall.
- `coverage` in the JSON stats is the expected acceptance ceiling *on that corpus*; real traffic elsewhere
  can differ a lot (that gap is what the SVD context-aware selector in the vLLM thread addresses).

## Layout & provenance

- `src/tf_util/` — installable package (`uv tool install`); `tf-util` is its console script.
- `draft_vocab.py` / `merge_draft_vocab.py` — self-contained PEP 723 copies of `tools/draft_vocab.py`
  (modified) and `tools/merge_draft_vocab.py` in the TensorFold repo (the repo itself is back to stock),
  for shipping to machines where the package isn't installed.
- The two implementations were verified to produce byte-identical output.
