# tensorfold-utils

Standalone helpers for building **draft vocabularies** for [TensorFold](https://github.com/ashhart/TensorFold)
and the related vLLM work ([PR #59740](https://github.com/vllm-project/vllm/pull/59740)). They live outside
the main repo on purpose: count a corpus wherever the text is, then merge the counts anywhere.

**Always update README.md, if needed, before git add and commit.**

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

## Commands

### `tf-util make-vocab` — count a corpus

```bash
tf-util make-vocab TOKENIZER_JSON -p PATH [-p PATH ...] [-o OUT.json]
```

`-p` accepts a file, a directory (walked recursively), or a glob; repeatable. Tokenizes every matched
file in full (no size limit) and writes a per-host counts JSON — default `<hostname>.json`
(e.g. `vyper.json`), override with `-o`. The JSON has `host`, `count_files`, `count_tokens`, `counts`.
Counting only: all selection limits live in `merge`.

### `tf-util merge` — sum counts, select vocab

```bash
tf-util merge TOKENIZER_JSON OUT.txt --size 32768 --min-count 10 a.json b.json ... [--keep-below 1024] [--added-tokens]
```

Sums the counts files, then keeps all IDs below `--keep-below` (default 1024), optionally the tokenizer's
added tokens, then frequency-ranked IDs down to `--min-count`, padded to `--size` with the lowest unused
IDs. Merging is plain count addition, so the result equals a single run over the union corpus.

## Workflow: corpus split across machines

1. Copy `draft_vocab.py` and the model's `tokenizer.json` to each machine holding text. **The tokenizer
   revision must be identical everywhere** — mismatched vocab IDs corrupt the counts.
2. On each machine:
   ```bash
   tf-util make-vocab tokenizer.json -p code/ -o local.json
   ```
3. Ship the `local.json` files back (any channel).
4. Merge once, deciding size/min-count a single time:
   ```bash
   tf-util merge tokenizer.json draft_vocab.txt --size 32768 --min-count 10 *.json
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
