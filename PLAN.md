# Plan: Audio Language Detection (Danish vs English)

This document describes the experimental plan for building and evaluating audio language
detection systems, scaling from an M4 Max laptop down to Raspberry Pi devices and
ultimately Bang & Olufsen headsets.

## Summary

We are building a binary classifier (Danish vs English) for audio. The system will be
evaluated on speaker-independent test data (1h Danish + 1h English, configurable), with
accuracy reported per utterance-length group. We start simple and progressively try more
complex methods from the literature survey.

## Code Organisation

```text
src/tiny_language_detection/   # Core modules (imported, not executed)
├── data/                       # Data loading and sampling
│   ├── common_voice.py         # CV26 loader, speaker-aware sampling
│   └── preprocessing.py        # Audio preprocessing (resample, mono, etc.)
├── features/                   # Feature extraction
│   ├── mfcc.py                 # MFCC extraction
│   └── spectrogram.py          # Log Mel-spectrogram, wavelet (later)
├── models/                     # Model architectures
│   ├── cnn.py                  # Simple CNN baselines
│   ├── rnn.py                  # CNN-RNN hybrids
│   └── transformers.py         # Compressed transformers (later)
├── eval/                       # Evaluation utilities
│   ├── metrics.py              # Accuracy, per-group metrics
│   └── speech_detection.py     # Ensure speech presence in clips
└── config/                     # Configuration (sample rate, etc.)
    └── defaults.py             # Sensible defaults from literature

src/scripts/                    # Executable scripts (uv run)
├── sample_data.py              # Create balanced test/train splits
├── train.py                    # Train a model
├── evaluate.py                 # Evaluate and export metrics
└── generate_dashboard.py       # Create HTML results dashboard

data/                           # Raw and processed data
├── cv26-da.tar.gz              # Danish Common Voice 26 (present)
├── cv26-en.tar.gz              # English Common Voice 26 (TBD)
├── sampled/                    # Sampled audio files (1h DA + 1h EN)
└── experiments/                # Experiment results (JSON, logs)

config/                         # Optional YAML configs
└── experiments.yaml            # Experiment definitions

results/                        # Generated HTML dashboard, plots
└── dashboard.html
```text

## Experiment Phases

We progress from simple to complex. Each phase produces comparable metrics so we can
trade off accuracy vs. compute cost.

### Phase 1: MFCC + Simple CNN Baseline

**Method:** Extract MFCCs (13-40 coefficients, 25ms window, 10ms hop) and train a
lightweight CNN (2-3 conv layers + global average pooling + dense head).

**Why:** MFCCs are fast, low-memory, and well-suited for edge. Simple CNNs are easy to
compress later. See Liu (2026), Darvishi (2026), Patil et al. (2024) in literature
survey.

**Target accuracy:** Establish baseline (expect ~85-95% on clean CV26 data).

### Phase 2: Log Mel-Spectrogram + CNN-RNN

**Method:** Use Log Mel-spectrograms (64-128 bins) with a shallow CNN followed by a
small GRU/LSTM (1-2 layers, 64-128 hidden units).

**Why:** Temporal modelling may help with longer utterances. Good trade-off per Cerna et
al. (2023), Ezilarasan et al. (2026).

**Target:** Improve over Phase 1, especially on longer utterances.

### Phase 3: Wavelet Features (Optional Exploration)

**Method:** Use Discrete Wavelet Transform (DWT) or Continuous Wavelet Transform (CWT)
as features, potentially with a spiking or lightweight CNN.

**Why:** Higher accuracy reported by Fahim et al. (2025), Bin Liu et al. (2026), but at
greater compute cost. Assess if worth the trade-off.

**Target:** Determine if wavelet features justify extra complexity.

### Phase 4: Model Compression and Optimisation

**Techniques:**

- **Pruning:** Remove low-weight connections (Mou and Milanova, 2024)
- **Quantisation:** INT8 or mixed-precision (Bittner et al., 2025)
- **Knowledge distillation:** Train small student from large teacher (Bhati et al.,
  2025)
- **Dynamic quantisation:** Per-layer optimisation (Mekonnen et al., 2025)

**Why:** Essential for edge deployment. Measure accuracy drop vs. speedup on Pi
hardware.

**Target:** 5-10x size reduction with <2% accuracy loss.

### Phase 5: Compressed Transformers (Exploratory)

**Method:** Token compression, margin-based contrastive learning, layer-wise optimisation
per Rezaabad et al. (2025), Jiang et al. (2025).

**Why:** State-of-the-art accuracy but complex. Assess feasibility for edge.

**Target:** Understand upper bound on accuracy.

## Data Sampling Strategy

### Speaker-Independent Splits

**Core requirement:** No speaker appears in both train and test sets.

**Approach:**

1. Parse speaker IDs from Common Voice metadata.
2. Group all clips by speaker.
3. Assign 80% of speakers to train, 20% to test (random split, stratified by language).
4. From test speakers, sample 1h Danish + 1h English (configurable).
5. From train speakers, sample matching distribution for training.

**Configuration:** Make split ratios and target durations configurable via CLI or YAML.

### Utterance Length Grouping

When building the test set, group utterances by duration:

| Group     | Duration range |
| --------- | -------------- |
| Short     | 0-2 s          |
| Medium    | 2-4 s          |
| Long      | 4-6 s          |
| Very long | 6+ s           |

Report accuracy separately for each group. This reveals model performance across
different amounts of available speech signal.

### Speech Presence Verification

For truncated clips (e.g., testing at 1s), verify actual speech is present:

1. Compute energy or zero-crossing rate.
2. Optionally use a lightweight VAD (Voice Activity Detector) like WebRTC VAD.
3. Discard clips with insufficient activity.

### Audio Configuration

**Defaults (adjustable via config):**

- Sample rate: 16 kHz (common in literature, balances quality and size)
- Channels: Mono (downmix if stereo)
- Bit depth: 16-bit PCM
- Format: WAV or FLAC (lossless for consistency)

**Rationale:** 16 kHz captures speech frequencies adequately (up to 8 kHz Nyquist),
widely used in edge audio work (Liu, 2026; Patil et al., 2024).

## No Unit Tests

This is an experimental repository for rapid prototyping. We prioritise iteration speed
over robustness. Validate experiments manually via the evaluation scripts and dashboard.

## Model Architectures to Try

| Phase | Architecture             | Params (approx) | Lit. refs                   |
| ----- | ------------------------ | --------------- | --------------------------- |
| 1     | MFCC + 3-layer CNN       | 50-100k         | Liu (2026), Darvishi (2026) |
| 2     | Spec + CNN-GRU           | 100-300k        | Cerna et al. (2023)         |
| 3     | DWT + CNN                | 100-200k        | Fahim et al. (2025)         |
| 4a    | Pruned Phase 1 model     | 10-30k          | Mou and Milanova (2024)     |
| 4b    | Quantised Phase 1 model  | 50-100k (INT8)  | Bittner et al. (2025)       |
| 5     | Compressed transformer   | 1-5M            | Rezaabad et al. (2025)      |

## Evaluation Methodology

### Metrics

| Metric                  | Description                                |
| ----------------------- | ------------------------------------------ |
| Overall accuracy        | % correct predictions on full test set     |
| Accuracy by duration    | Separate accuracy for each length group    |
| Per-language accuracy   | Danish vs English breakdown                |
| Confusion matrix        | FP/FN rates (useful for error analysis)    |
| Inference latency       | Mean/median time per clip (on target HW)   |
| Model size              | On-disk size and RAM usage                 |
| Energy (later)          | Joules per inference (if measurable)       |

### Evaluation Protocol

1. **Training:** Train on train split only (no leakage from test speakers).
2. **Validation:** Use 10% of train speakers for early stopping / hyperparameter tuning.
3. **Testing:** Evaluate on held-out test speakers only.

### Reporting

All metrics exported to JSON in `data/experiments/<experiment_id>/metrics.json`, with
plots and aggregate tables in the HTML dashboard.

## Hardware Evaluation Plan

We evaluate latency and feasibility at each hardware tier.

### Tier 1: M4 Max Laptop (Development)

**Purpose:** Rapid prototyping, full model training, debugging.

**Benchmarks:**

- Training time per model.
- Inference latency (CPU and GPU, if applicable).
- Establish performance ceiling.

### Tier 2: Raspberry Pi 5

**Purpose:** First edge target, moderate compute (~4x Cortex-A76, 4-8 GB RAM).

**Benchmarks:**

- Inference latency (single-threaded and multi-threaded).
- RAM usage.
- Quantisation speedup.

**Tooling:** Use `time` command, optional profiling with `perf` or `py-spy`.

### Tier 3: Raspberry Pi Zero (or Zero 2 W)

**Purpose:** Constrained edge target (single-core or quad-core ARM, 512 MB RAM).

**Benchmarks:**

- Inference latency (may be real-time only for smallest models).
- RAM usage (critical constraint).
- Model size (must fit in available storage).

### Tier 4: Bang & Olufsen Headsets

**Purpose:** Ultimate deployment target.

**Challenges:**

- Unknown chipset (likely proprietary, possibly Qualcomm or similar).
- May require custom firmware or companion app.
- Likely severe memory and compute constraints.

**Approach:**

1. Identify chipset and available SDKs.
2. Profile smallest viable model (Phase 4 compressed).
3. Consider TinyML frameworks (Edge Impulse, acoupi) per Patil et al. (2024),
   Vuilliomenet et al. (2025).

## Experiment Results Organisation

### Directory Structure

```text
results/
├── dashboard.html              # Main interactive dashboard
├── experiments/
│   ├── exp001_mfcc_cnn/
│   │   ├── metrics.json        # Machine-readable results
│   │   ├── plots/              # PNG plots (accuracy by duration, etc.)
│   │   ├── logs/               # Training logs
│   │   └── model/              # Saved model weights
│   ├── exp002_cnn_rnn/
│   └── ...
└── summaries/
    └── comparison_table.json   # Aggregated comparison across experiments
```text

### HTML Dashboard

**Features:**

- Interactive plots (Plotly or Altair) showing accuracy by duration group.
- Comparison table of all experiments (accuracy, latency, model size).
- Per-experiment drill-down with confusion matrices, sample predictions.
- Filter by hardware tier (show only Pi-relevant results).
- Export to CSV.

**Generation:** `uv run src/scripts/generate_dashboard.py` aggregates all
`metrics.json` files and produces `dashboard.html`.

## Timeline and Milestones

| Week | Milestone                                       | Deliverable                        |
| ---- | ----------------------------------------------- | ---------------------------------- |
| 1-2  | Data sampling scripts, Phase 1 baseline         | `sample_data.py`, MFCC+CNN model  |
| 3    | Phase 2 (CNN-RNN), evaluation framework         | `evaluate.py`, metrics w/ groups  |
| 4    | Dashboard v1, Pi 5 benchmarks                   | `dashboard.html`, latency report  |
| 5    | Phase 3 (wavelets, optional), compression tests | Compressed models, size/acc tradeoff |
| 6    | Pi Zero benchmarks, headset discovery           | Headset chipset report           |
| 7    | Phase 5 (transformers, exploratory)             | Transformer benchmark              |
| 8    | Final dashboard, documentation                  | Complete results                   |

## Risks and Mitigations

| Risk                            | Mitigation                                               |
| ------------------------------- | -------------------------------------------------------- |
| English CV26 data unavailable   | Use Common Voice 25 or alternative corpus (e.g., VoxLingua) |
| Headset SDK not available       | Partner with B&O or use proxy hardware with similar specs |
| Real-time not achievable on Pi  | Use aggressive compression, quantisation, smaller windows |
| Speech-presence check too slow  | Use simple energy threshold, skip if proven unnecessary   |

## Getting Started

1. **Install dependencies:** `make install`
2. **Sample data:** `uv run src/scripts/sample_data.py`
3. **Train baseline:** `uv run src/scripts/train.py --phase 1`
4. **Evaluate:** `uv run src/scripts/evaluate.py --phase 1`
5. **Generate dashboard:** `uv run src/scripts/generate_dashboard.py`
