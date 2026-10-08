# tensorfold-utils

Standalone helpers for building **draft vocabularies** for [TensorFold](https://github.com/ashhart/TensorFold)
and the related vLLM work ([PR #59740](https://github.com/vllm-project/vllm/pull/59740)). They live outside
the main repo on purpose: count a corpus wherever the text is, then merge the counts anywhere.

**Always update README.md, if needed, before git add and commit.**
**Always bump version (pyproject.toml) before git add and commit.**

## How to update AGENTS.md

It is not a worklog or decision tracker. It is only to reflect the current state of the code base and
provide fast access to entrypoints to future agents. Keep it slim and concise at all times.

## Background

In speculative/MTP decoding the draft model proposes tokens and the target model verifies them with its
full vocabulary. So the draft head only needs to score a *subset* of the vocab: a token missing from the
list can't be proposed, which costs a little acceptance length — never correctness (output stays
byte-identical to undrafted decoding). A smaller list means proportionally fewer lm_head rows to read,
which speeds up decode on bandwidth-bound hardware (DGX Spark, Macs). The list is built by ranking token
frequencies over a public text corpus, always keeping low IDs and special tokens.

## Layout

- `src/tf_util/` — installable package; `uv tool install git+https://github.com/eleqtrizit/tensorfold-utils`
  puts `tf-util` on PATH. Its console script dispatches `make-vocab` → `tf_util.draft_vocab.main`,
  `merge` → `tf_util.merge_draft_vocab.main`.
- `draft_vocab.py` / `merge_draft_vocab.py` — self-contained PEP 723 copies of the same logic
  (`tokenizers==0.22.2` pinned, Python ≥ 3.11; originals: `tools/draft_vocab.py` modified and
  `tools/merge_draft_vocab.py` in the TensorFold repo), for shipping to machines without the package.
  Each inlines its own `resolve_tokenizer` (shelling out to the `hf` CLI) so HF-id resolution works
  without the package; the shared copy lives in `src/tf_util/hf_tokenizer.py`.

## Commands

### `tf-util make-vocab` — count a corpus

```bash
tf-util make-vocab TOKENIZER -p PATH [-p PATH ...] [-o OUT.json]
```

`TOKENIZER` is a local `tokenizer.json` path **or** a Hugging Face repo id (`org/model`,
`org/model@revision`, or a bare id like `gpt2`). For a repo id, `hf_tokenizer.resolve_tokenizer`
downloads just `tokenizer.json` via the `hf` CLI into `$TF_TOKENIZER_CACHE` (default
`/tmp/tokenizers`)/`<repo>`/`tokenizer.json` and reuses the cached file; pin a revision with
`@revision` for byte-reproducible counts.

`-p` accepts a file, a directory (walked recursively), or a glob; repeatable. `-e`/`--exclude` takes a
substring: any file whose path contains it is skipped (repeatable, e.g. `-e .venv -e node_modules`).
A built-in `ALWAYS_EXCLUDE` list is applied in the background on top of `-e` (VCS internals,
node_modules/venv-like caches, build outputs, IDE/agent dirs like `.claude`/`.pi`/`.agents`/`.codex`,
logs/temp/secrets — full list in `ALWAYS_EXCLUDE` in `draft_vocab.py`); excluded dirs are pruned so
`os.walk` never even descends into them. A cheap binary sniff (NUL byte / magic numbers in the
first 8KB, git's heuristic) rejects non-text files before decoding; see `looks_like_text`. Skipped
files are counted (`"skipped"` in the JSON) and listed with reasons via `-v/--verbose`.
Tokenizes every matched
file in full (no size limit) and writes a per-host counts JSON — default `<hostname>.json`
(e.g. `vyper.json`), override with `-o`. The JSON has `host`, `count_files`, `count_tokens`, `counts`.
Counting only: all selection limits live in `merge`. A live status line (braille spinner + running
files/tokens + current path) is drawn on stderr while parsing, then cleared for the final JSON;
auto-disabled when stderr isn't a TTY.

### `tf-util merge` — sum counts, select vocab

```bash
tf-util merge TOKENIZER a.json b.json ... [--out draft_vocab.txt] [--size 32768] [--min-count 10] [--keep-below 1024] [--added-tokens]
```

`TOKENIZER` accepts a local path or an HF repo id (same resolution as `make-vocab`). Writes the
newline-separated ID list to `--out` (default `draft_vocab.txt`). Both commands print a JSON
one-liner to stdout with a `"wrote"` key naming every file written plus stats.

Sums the counts files, then keeps all IDs below `--keep-below` (default 1024), optionally the tokenizer's
added tokens, then frequency-ranked IDs down to `--min-count`, padded to `--size` with the lowest unused
IDs. Merging is plain count addition, so the result equals a single run over the union corpus.

## Workflow: corpus split across machines

1. On each machine (installed tool or standalone script) just name the model — no tokenizer copy needed:
   ```bash
   tf-util make-vocab org/model@revision -p code/ -o local.json
   ```
   Pin `@revision` for byte-reproducible counts; `hf` must be on PATH (`HF_TOKEN` for gated repos).
2. Ship the `local.json` files back (any channel).
3. Merge once, deciding size/min-count a single time (writes `draft_vocab.txt` by default):
   ```bash
   tf-util merge org/model@revision --size 32768 --min-count 10 *.json
   ```

The output format matches the engine's expectation (plain newline-separated integer IDs; runtime pads to
a multiple of 64 itself), so the result can slot into TensorFold's `draft_ids()` / a vLLM
`speculative_config.draft_token_map` / SGLang `--speculative-token-map` style list.

## Caveats

- No file-size limit: every file matched by `-p` is read in full — watch for generated blobs.
- Corpus choice is the quality lever: a CPython-stdlib-built list is code/English-skewed and loses
  acceptance on other languages. Build from traffic that resembles yours, and hash your inputs
  (`sha256sum` tokenizer + corpus manifest) if reproducibility matters.
- `coverage` in the merge output is the expected acceptance ceiling *on that corpus*; real traffic
  elsewhere can differ a lot (that gap is what the SVD context-aware selector in the vLLM thread addresses).
