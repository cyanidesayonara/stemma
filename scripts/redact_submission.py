"""Print or save a Partner Center submission JSON without its upload link.

A draft's ``FileUploadUrl`` is a write-capable Azure blob SAS link: anyone
holding it can replace the package and images the Store ingests when the
draft is committed. This repository's Actions logs and artifacts are
public, so every submission JSON the Store workflow shows goes through
here first. The key is removed, and any other blob SAS link in the JSON is
replaced as well.

    python scripts/redact_submission.py submission.json            # prints
    python scripts/redact_submission.py submission.json --out x.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.store_listing import parse_msstore_submission_json  # noqa: E402

REDACTED = "<redacted: write-capable upload link>"
SAS_LINK = re.compile(r"https://[^\"\s]*\.blob\.core\.windows\.net/[^\"\s]*",
                      re.I)


def redact(value):
    """A copy of *value* with upload links removed, recursively."""
    if isinstance(value, dict):
        return {
            key: (REDACTED if key.lower() == "fileuploadurl" else redact(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return SAS_LINK.sub(REDACTED, value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission", type=Path,
                        help="msstore submission output or a JSON file")
    parser.add_argument("--out", type=Path,
                        help="write here instead of printing")
    args = parser.parse_args(argv)
    raw = args.submission.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = parse_msstore_submission_json(raw)
    text = json.dumps(redact(data), indent=2) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8", newline="\n")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
