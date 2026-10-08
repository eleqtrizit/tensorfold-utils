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
tf-util make-vocab TOKENIZER_JSON -p PATH [-p PATH ...] [-o OUT.json]

# Sum counts files, then select the vocab (all limits live here)
tf-util merge TOKENIZER_JSON OUT.txt [--size 32768] [--min-count 10] \
        [--keep-below 1024] [--added-tokens] a.json b.json ...
```

`-p` accepts a file, a directory (walked recursively), or a glob; repeat it for multiple paths.
There is no `--max-bytes` limit anymore — every matching file is read in full.

### `make-vocab`

Alias: `uv run draft_vocab.py ...` (self-contained PEP 723 script; logic in `src/tf_util/draft_vocab.py`).

Tokenizes the corpus files given via `-p` (repeatable: files, directories, or globs) and writes the
raw per-ID counts as a portable, mergeable JSON file named after the host by default (`vyper.json`),
including a `host` key. **This is the counting half of the split workflow** — no size/limit decisions
happen here; those all happen at merge time.

### `merge`

Alias: `uv run merge_draft_vocab.py ...`, logic in `src/tf_util/merge_draft_vocab.py`.

Sums the counts from one or more counts files and selects the draft vocabulary: all IDs below
`--keep-below` (default 1024), optionally all tokenizer added tokens (`--added-tokens`), then
frequency-ranked IDs down to `--min-count`, padded to `--size` with the lowest unused IDs. The result
is **byte-identical** to counting the union corpus in a single run (verified with `cmp`). The counts
files are small (one entry per appearing ID, KBs–MBs), so they ship easily.

The JSON stats include `coverage` — the fraction of corpus tokens covered by the selected vocab, i.e. the
expected acceptance ceiling *on that corpus*.

## Workflow: corpus split across machines

1. Copy `draft_vocab.py` and the model's `tokenizer.json` to each machine holding text.
   **The tokenizer revision must be identical everywhere** — mismatched vocab IDs corrupt the counts.
2. On each machine (installed tool or the standalone script):
   ```bash
   tf-util make-vocab tokenizer.json -p code/ -o local.json
   ```
3. Ship the `local.json` files back (any channel).
4. Merge once, deciding size/min-count a single time:
   ```bash
   tf-util merge tokenizer.json draft_vocab.txt --size 32768 --min-count 10 *.json
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
