"""tf-util — Create draft vocabulary on your own code bases to speed up TensorFold

Example for GLM 5.3 Flash (https://huggingface.co/zai-org/GLM-5.3-Flash).  Run on as many machines as you want:

    tf-util make-vocab zai-org/GLM-5.3-Flash -p PATH1 -p PATH2 [-e .venv -e node_modules]

then merge all results:

    tf-util merge zai-org/GLM-5.3-Flash <json file 1> <json file 2> ...

TOKENIZER is a local tokenizer.json path OR a Hugging Face repo id
(`org/model`, `org/model@revision`, or a bare id like `gpt2`); a repo id is
fetched via the `hf` CLI into $TF_TOKENIZER_CACHE (default /tmp/tokenizers)/<repo>/.

  tf-util make-vocab TOKENIZER -p PATH [-p PATH ...] [-e STRING ...] [-o OUT.json]    count a corpus; writes <hostname>.json
  tf-util merge TOKENIZER COUNTS.json [--out OUT.txt] [--size N] [--min-count N] [--keep-below N] [--added-tokens]
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
