"""Retry paired-statistics export with NumPy scalar JSON conversion."""
from __future__ import annotations

import numpy as np

import compute_final_paired_statistics as target


_original_dumps = target.json.dumps


def _default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _dumps(*args, **kwargs):
    kwargs.setdefault("default", _default)
    return _original_dumps(*args, **kwargs)


def main() -> None:
    target.json.dumps = _dumps
    try:
        target.main()
    finally:
        target.json.dumps = _original_dumps


if __name__ == "__main__":
    main()
