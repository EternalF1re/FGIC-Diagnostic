"""Retry final aggregation with UTF-8 as the explicit text-read default."""
from __future__ import annotations

from pathlib import Path

import aggregate_formal


_original_read_text = Path.read_text


def _read_text_utf8(self: Path, encoding=None, errors=None):
    return _original_read_text(
        self,
        encoding="utf-8" if encoding is None else encoding,
        errors=errors,
    )


def main() -> None:
    Path.read_text = _read_text_utf8
    try:
        aggregate_formal.main()
    finally:
        Path.read_text = _original_read_text


if __name__ == "__main__":
    main()
