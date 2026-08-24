# Contributing

Please open an issue before changing a frozen scientific protocol. Bug fixes should include a regression test and state whether they alter previously reported values.

Install development tools with `python -m pip install -e .[dev]`, then run:

```bash
python -m pytest -q
python -m ruff check src tests
```

Do not commit datasets, checkpoints, OOF arrays, local dataset configs, logs or credentials. New experiment outputs must use a new directory and must not overwrite a completed run.

