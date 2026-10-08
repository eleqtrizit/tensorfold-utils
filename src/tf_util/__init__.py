"""tf-util — single entry point for the TensorFold draft-vocabulary helpers.

  tf-util make-vocab TOKENIZER_JSON -p PATH [-p PATH ...] [-o OUT.json]
  tf-util merge TOKENIZER_JSON OUT.txt [--size N] [--min-count N] [--keep-below N] [--added-tokens] COUNTS.json ...

  make-vocab -> tf_util.draft_vocab        (count a corpus; writes <hostname>.json counts)
  merge      -> tf_util.merge_draft_vocab  (sum counts files, apply size/min-count limits)
"""

from __future__ import annotations

import sys

from tf_util import draft_vocab, merge_draft_vocab

COMMANDS = {
    "make-vocab": draft_vocab.main,
    "merge": merge_draft_vocab.main,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip(), file=sys.stderr)
        return 0 if argv else 2
    sub, rest = argv[0], argv[1:]
    if sub not in COMMANDS:
        print(f"tf-util: unknown command {sub!r}; expected one of {', '.join(COMMANDS)}",
              file=sys.stderr)
        return 2
    return COMMANDS[sub](rest)


if __name__ == "__main__":
    sys.exit(main())
