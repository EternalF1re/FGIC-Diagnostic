"""Run the frozen Task-A finalizer with deterministic UTF-8 text decoding."""

from pathlib import Path


_read_text = Path.read_text


def _read_text_utf8(self, encoding=None, errors=None):
    return _read_text(self, encoding=encoding or "utf-8", errors=errors)


Path.read_text = _read_text_utf8

import finalize


if __name__ == "__main__":
    finalize.main()
