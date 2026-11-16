# Contributing

Thank you for your interest in WarfaRisk. This is a research project, so changes are reviewed with care.

## Before you open a pull request

```bash
uvx ruff check . && uvx ruff format --check .
uv run --with pytest,flask,flask-cors,catboost,pandas pytest
```

Both commands must pass.

## Guidelines

- Keep changes small and focused on one problem.
- Do not commit patient-level data, API keys or `.env` files. The `data/` folder holds split ID files only.
- If a change affects a reported number, update `docs/RESULTS.md` and say which script and output file support it.
- The code is released under the Apache License 2.0, and your contribution will be too.
