# Tiny Language Detection

Audio language detection for edge devices. The project is part of REINS and focuses on
small speech language-identification models with a practical deployment target around
100 KB.

## Current goals

- Build a Pareto curve for model sizes from roughly 50 KB to 1 MB.
- Prioritise dense experiments around 50-200 KB, especially near 100 KB.
- Evaluate accuracy, macro-F1, per-language accuracy, latency, RAM, and model size.
- Compare a focused major-European-language track with a broad 23-language track.
- Support configurable candidate language sets at inference time.
- Avoid treating restricted language selection as only selected-logit renormalisation;
  the full language distribution can help restricted classification.

## Experiment language tracks

Experiments should compare two language-coverage tracks:

1. **Major European languages** — the current focused deployment target:
   - English, covering British and American English
   - Danish
   - German
   - Dutch
   - French
   - Spanish
   - Italian
2. **Broad European coverage** — all 23 languages available in the YODAS-Granary
   language-detection dataset.

The 23-language track is worth testing because broad coverage is better if a compact
model can maintain strong performance. If the focused track performs substantially
better, it may be the more useful deployment choice.

Chinese is out of scope for now because it is not part of the YODAS-Granary dataset.

Before describing the broad track externally as "official EU languages", verify that the
23-language YODAS-Granary inventory exactly matches the intended EU-language list.

## Configurable candidate languages

Users should be able to restrict the languages considered at inference time, for example
to English+Danish. Restricted classification should ideally perform better than full-set
classification.

When implementing this, include naive selected-logit renormalisation only as a baseline.
Prefer approaches that can use full logits or embeddings together with a
selected-language mask, such as:

- a subset-aware decision head
- small subset-specific calibration heads for common subsets

Keep the open-set design question explicit: if the user selects English+Danish and the
input is actually German, the system may need either forced-choice behaviour or a
`none_of_the_selected_languages` option.

## Stack

- Python 3.12+
- `uv` package manager
- Pytest for testing
- Ruff for linting and formatting

## Layout

- `src/tiny_language_detection/` — Core package with detection logic and models
- `src/scripts/` — Executable scripts, run with `uv run`
- `data/` — Dataset files, often large and gitignored
- `results/` — Experiment results and Pareto/frontier outputs
- `docs/` — Experiment notes and background documentation
- `.github/workflows/` — CI/CD pipelines
- `PLAN.md` — Current model-size and configurable-language plan

## Running it

### Install

```bash
make install
```

### Run demo

```bash
uv run src/scripts/demo_server.py
```

Server runs at <http://localhost:7860>.

### Train model

```bash
# Train CNN
uv run src/scripts/train_cnn.py \
  --dataset saattrupdan/yodas-granary-language-detection \
  --use-hf

# Train CNN-RNN
uv run src/scripts/train_cnn_rnn.py \
  --dataset saattrupdan/yodas-granary-language-detection \
  --use-hf
```

### Lint and check

```bash
make check
```

## Conventions

- Use `uv run` for Python scripts. Do not activate a virtual environment manually or run
  scripts with bare `python`.
- Put reusable package code in `src/tiny_language_detection/`.
- Put executable experiment or utility scripts in `src/scripts/`.
- Use type hints for Python 3.12+.
- Use Google-style docstrings.
- Use British English in comments, docstrings, and documentation, for example
  "labelled", "colour", and "analyse".
- Use relative imports inside package modules and absolute imports in scripts.

## Gotchas

- **Data files are large** — The `data/` directory may contain large audio files or
  datasets. Check `.gitignore` before adding anything there.
- **Model size matters** — Do not optimise only for accuracy. Track size, latency, and
  memory for each experiment.
- **Restricted-language inference is non-trivial** — Do not silently implement it as
  selected-logit renormalisation without also recording it as a baseline and comparing
  stronger methods.
- **Quantisation trade-offs** — Quantisation may reduce storage size without reducing
  runtime RAM if weights are dequantised for CPU inference.

## Phase 1 results

**Baseline:** MFCC features with a 3-layer CNN for Danish-English classification.

| Metric | Value |
| --- | ---: |
| Overall accuracy | 81.7% |
| Danish accuracy | 86.9% |
| English accuracy | 75.1% |

**Test set:** 1,729 clips with around one hour each of Danish and English speech,
speaker-independent across 544 speakers.

**Key findings:**

- Short clips from 0-2s perform best at around 86% accuracy.
- Longer clips are stable around 80-81% accuracy.
- There is an 11.8 percentage-point gap between Danish and English accuracy.
- Training accuracy was 91.6% and test accuracy was 81.7%, indicating mild overfitting.
