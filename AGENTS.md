# Tiny Language Detection

Audio language detection for edge devices, distinguishing between Danish and English
speech. Part of the REINS project.

## Stack

- Python 3.12+
- `uv` package manager
- Pytest for testing
- Ruff for linting and formatting

## Layout

- `src/tiny_language_detection/` — Core package with detection logic and models
- `src/scripts/` — Executable scripts (run with `uv run`)
- `data/` — Dataset files (may be large, often gitignored)
- `tests/` — Test suite
- `.github/workflows/` — CI/CD pipelines
- `.devcontainer/` — Development container configuration

## Running it

Install dependencies:

```bash
make install
```

Run scripts:

```bash
uv run src/scripts/<script_name>.py
```

Run tests:

```bash
make test
```

Run linters and type checkers:

```bash
make check
```

## Conventions

See `README.md` for detailed Python conventions covering:

- Code organisation (modules in `src/tiny_language_detection/`, scripts in
  `src/scripts/`)
- Type hints (Python 3.12+ syntax)
- Documentation (Google-style docstrings)
- Imports (relative in modules, absolute in scripts)

## Gotchas

- **Data files are large** — The `data/` directory may contain large audio files or
  datasets. Check `.gitignore` before adding anything there.
- **Use `uv run`** — Always execute scripts with `uv run`, never activate a virtual
  environment or use `python -m`.
- **British English** — Comments, docstrings, and documentation use British English
  (e.g., "labelled", "colour", "analyse").
