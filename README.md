# Tiny Language Detection

Audio language detection for edge devices, developed as part of the REINS project.

The project trains compact speech language-identification models and evaluates the
trade-off between model size, runtime cost, and classification quality. The deployment
target is ideally around 100 KB, with experiments spanning roughly 50 KB to 1 MB.

## Target languages

The main target language set is:

- English, covering British and American English
- Danish
- German
- Dutch
- French
- Spanish
- Italian
- Chinese

The broader YODAS-Granary dataset used by the project contains 23 languages, but
current experiments should focus on the target set above unless the experiment
explicitly studies broader multilingual training.

## Configurable candidate languages

Inference should support user-selected candidate language sets. For example, a user may
want the model to choose only between English and Danish, even if the base model was
trained with more languages.

Restricted candidate sets should improve classification performance relative to the full
language set. The implementation should not only discard logits for non-selected
languages and renormalise the remaining logits. The full language distribution may
contain useful information for a restricted decision, such as German or Dutch evidence
helping distinguish Danish from English.

Planned decision methods are:

- selected-logit renormalisation as a simple baseline
- a subset-aware decision head using full logits or embeddings plus a candidate-language
  mask
- tiny subset-specific calibration heads for common subsets such as English+Danish

See [`PLAN.md`](PLAN.md) for the full experiment plan.

## Model-size Pareto sweep

Experiments should produce a Pareto curve across models from roughly 50 KB to 1 MB.
Use 50 KB intervals as the nominal target, but allow practical deviations because model
sizes change discretely with architecture and quantisation choices.

Prioritise dense coverage around 50-200 KB, especially near the ideal 100 KB deployment
target.

For each model, report:

- model size on disk
- number of parameters
- quantisation method
- latency on target or representative hardware
- peak RAM usage, where practical
- accuracy and macro-F1
- per-language accuracy
- confusion matrix

## Dataset

The current training data is based on the Hugging Face dataset
`saattrupdan/yodas-granary-language-detection`, derived from `espnet/yodas-granary`.

The repository also documents a balanced YODAS-Granary test subset:

| Split | Samples | Duration | Notes |
| --- | ---: | ---: | --- |
| Train | 211k | 415h | Parquet shards |
| Test | 2,300 | 8.5h | Balanced across 23 languages |

## Installation

```bash
make install
```

The project uses Python 3.12+ and `uv` for dependency management.

## Usage

Run the demo server:

```bash
uv run src/scripts/demo_server.py
```

Train a compact CNN:

```bash
uv run src/scripts/train_cnn.py \
  --dataset saattrupdan/yodas-granary-language-detection \
  --use-hf
```

Train a CNN-RNN model:

```bash
uv run src/scripts/train_cnn_rnn.py \
  --dataset saattrupdan/yodas-granary-language-detection \
  --use-hf
```

Generate a Pareto plot:

```bash
uv run src/scripts/plot_pareto.py
```

Run quality checks:

```bash
make check
```

## Current baseline

Phase 1 used MFCC features with a 3-layer CNN for Danish-English classification.

| Metric | Value |
| --- | ---: |
| Overall accuracy | 81.7% |
| Danish accuracy | 86.9% |
| English accuracy | 75.1% |

The test set contained 1,729 speaker-independent clips, with around one hour each of
Danish and English speech. See `docs/phase1-mfcc-cnn-baseline.md` if available in the
working copy, and `docs/experiments.md` for the current experiment overview.

## Documentation

- [`PLAN.md`](PLAN.md) — current target-size and configurable-language plan
- [`docs/experiments.md`](docs/experiments.md) — experiment overview
- [`docs/literature-survey.md`](docs/literature-survey.md) — background survey

## License

See [`LICENSE`](LICENSE).
