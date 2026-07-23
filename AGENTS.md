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
- `results/` — Experiment results and HTML dashboard
- `.github/workflows/` — CI/CD pipelines

## Running it

### Install

```bash
make install
```

### Run Demo

```bash
make demo
```

Flask is auto-installed. Server runs at http://localhost:7860

### Train Model

```bash
# Train CNN (23 languages)
uv run src/scripts/train_cnn.py --dataset saattrupdan/yodas-granary-language-detection --use-hf

# Or train CNN-RNN
uv run src/scripts/train_cnn_rnn.py --dataset saattrupdan/yodas-granary-language-detection --use-hf
```

### Lint & Check

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

## Phase 1 Results

**Baseline:** MFCC (20 coeffs, 25ms/10ms window/hop) + 3-layer CNN (56k params)

| Metric            | Value |
|-------------------|-------|
| Overall accuracy  | 81.7% |
| Danish accuracy   | 86.9% |
| English accuracy  | 75.1% |

**Test set:** 1,729 clips (1h DA + 1h EN), speaker-independent (544 speakers).

Full experimental report in `docs/phase1-mfcc-cnn-baseline.md`.

**Key findings:**

- Short clips (0–2s) perform best (86%), longer clips stable (~80–81%)
- 11.8 pp gap between Danish and English accuracy
- Training accuracy 91.6%, test 81.7% — mild overfitting
