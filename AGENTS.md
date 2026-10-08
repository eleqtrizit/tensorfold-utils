# tensorfold-utils

Standalone helpers for building **draft vocabularies** for [TensorFold](https://github.com/ashhart/TensorFold)
and the related vLLM work ([PR #59740](https://github.com/vllm-project/vllm/pull/59740)). They live outside
the main repo on purpose: count a corpus wherever the text is, then merge the counts anywhere.

## Background

In speculative/MTP decoding the draft model proposes tokens and the target model verifies them with its
full vocabulary. So the draft head only needs to score a *subset* of the vocab: a token missing from the
list can't be proposed, which costs a little acceptance length — never correctness (output stays
byte-identical to undrafted decoding). A smaller list means proportionally fewer lm_head rows to read,
which speeds up decode on bandwidth-bound hardware (DGX Spark, Macs). The list is built by ranking token
frequencies over a public text corpus, always keeping low IDs and special tokens.

## Scripts

### `tf-util` — single entry point (dispatcher)

```bash
tf-util make-vocab ... # same args as draft_vocab.py
tf-util merge      ... # same args as merge_draft_vocab.py
```

`src/tf_util/` is an installable package (`uv tool install .` or a git URL). The `tf-util`
console script dispatches: `make-vocab` -> `tf_util.draft_vocab.main`, `merge` ->
`tf_util.merge_draft_vocab.main`. Logic lives in the package modules; the root-level scripts are
self-contained PEP 723 portable copies (verified byte-identical to the package). All commands take
the remaining arguments through verbatim.

All are PEP 723 uv scripts where standalone (`tokenizers==0.22.2` pinned, Python >= 3.11) — run with
`uv run <script> ...`, no venv or repo install needed. They are copies of `tools/draft_vocab.py`
(modified) and `tools/merge_draft_vocab.py` in the TensorFold repo; the repo itself is back to stock.

### `draft_vocab.py` — count and/or select (via `tf-util make-vocab`)

```bash
uv run draft_vocab.py TOKENIZER_JSON OUT.txt --size 32768 --min-count 10 'corpus/**/*.py' [--added-tokens] [--counts-out counts.json]
```

Tokenizes the corpus files and writes a sorted, one-ID-per-line draft vocabulary: all IDs below
`--keep-below` (default 1024), optionally all tokenizer added tokens, then frequency-ranked IDs down to
`--min-count`, padded to `--size` with the lowest unused IDs.

`--counts-out FILE` additionally dumps the raw per-ID counts as JSON. **For the split workflow this is
the important mode**: you only need the tokenizer + corpus here — the final `--size`/`--min-count`
decisions happen later at merge time.

### `merge_draft_vocab.py` — combine counts, then select (via `tf-util merge`)

```bash
uv run merge_draft_vocab.py TOKENIZER_JSON OUT.txt --size 32768 --min-count 10 a.json b.json ... [--added-tokens]
```

Sums the counts from one or more `--counts-out` files and applies exactly the same selection rules, so
the result is byte-identical to counting the union corpus in a single run (verified). The counts files
are small (one entry per appearing ID, KBs–MBs), so they ship easily.

## Workflow: corpus split across machines

1. Copy `draft_vocab.py` and the model's `tokenizer.json` to each machine holding text. **The tokenizer
   revision must be identical everywhere** — mismatched vocab IDs corrupt the counts.
2. On each machine:
   ```bash
   uv run draft_vocab.py tokenizer.json local.txt 'code/**/*.py' --counts-out local.json
   ```
3. Ship the `local.json` files back (any channel).
4. Merge once, deciding size/min-count a single time:
   ```bash
   uv run merge_draft_vocab.py tokenizer.json draft_vocab.txt --size 32768 --min-count 10 *.json
   ```

The output format matches the engine's expectation (plain newline-separated integer IDs; runtime pads to
a multiple of 64 itself), so the result can slot into TensorFold's `draft_ids()` / a vLLM
`speculative_config.draft_token_map` / SGLang `--speculative-token-map` style list.

## Caveats

- Files > 2 MB are silently skipped (`--max-bytes`) — remember this when "code everywhere" includes
  generated blobs; adjust `--max-bytes` or split patterns.
- Corpus choice is the quality lever: a CPython-stdlib-built list is code/English-skewed and loses
  acceptance on other languages. Build from traffic that resembles yours, and hash your inputs
  (`sha256sum` tokenizer + corpus manifest) if reproducibility matters — this was the repo's own pitfall.
- `coverage` in the JSON stats is the expected acceptance ceiling *on that corpus*; real traffic elsewhere
  can differ a lot (that gap is what the SVD context-aware selector in the vLLM thread addresses).
