"""Serialization-safe entry point for the immutable preflight implementation."""
import json
import numpy as np
import preflight

_dumps = json.dumps

def _default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

preflight.json.dumps = lambda obj, *args, **kwargs: _dumps(
    obj, *args, default=kwargs.pop("default", _default), **kwargs
)

if __name__ == "__main__":
    preflight.main()
