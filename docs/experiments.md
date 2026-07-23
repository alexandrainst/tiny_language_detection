# Experiments: Multilingual Language Detection

**Project:** Tiny Language Detection
**Repository:** <https://github.com/alexandrainst/tiny_language_detection>
**Dataset:** YODAS-Granary (23 languages, balanced test set)

---

## Goal

Find the optimal model architecture that maximises accuracy while minimising runtime RAM usage for edge deployment (earbuds: <1 MB, headphones: 1-2 MB).

We visualise this as a **Pareto frontier**: a scatter plot of RAM (x-axis) vs accuracy (y-axis), where Pareto-optimal models form the upper-left boundary (best accuracy for given RAM).

---

## Dataset

**YODAS-Granary** — 23-language speech dataset for edge language detection:

| Split | Samples | Duration | Format |
|-------|---------|----------|--------|
| Train  | 211k   | 415h     | Parquet shards (128 samples each) |
| Test   | 2,300  | 8.5h     | Balanced (100 per language) |

**Loading:** Direct download with `load_dataset()` (~28GB). No streaming required.

**HF Repo:** `saattrupdan/yodas-granary-language-detection`

**See also:** `docs/granary-dataset.md`

---

## Model Architectures

All models use **multi-class classification** (N languages, softmax output):

### CompactCNN

Scalable CNN with global average pooling (no RNN):

```python
# Tiny: 45k params, ~180 KB RAM
channels = [16, 32, 64]

# Small: 175k params, ~686 KB RAM (recommended)
channels = [32, 64, 128]

# Medium: 500k params, ~1.9 MB RAM
channels = [64, 128, 256]
```

**Best for:** Resource-constrained devices (earbuds, headphones)

### CNN-RNN

CNN encoder + unidirectional GRU:

```python
# Default: ~546k params, ~2.1 MB RAM
channels = [32, 64, 128]
hidden_size = 128
```

**Best for:** Scenarios where temporal modelling is critical

---

## Experiments

Results are tracked in `data/results.jsonl`. To visualise:

```bash
uv run src/scripts/plot_pareto.py --highlight phase4b-small-23
```

This generates `results/pareto_frontier.png` showing all experiments with the Pareto frontier.

### Planned Experiments

**Priority 1: Earbud Targets (<1 MB RAM)**

| Model ID | Architecture | Params | RAM | Storage | Training | Status |
|----------|-------------|--------|-----|---------|----------|--------|
| `tiny-direct` | CompactCNN Tiny | 45k | 180 KB | 180 KB | Direct | TODO |
| `small-direct` | CompactCNN Small | 175k | 686 KB | 686 KB | Direct | **TODO** |
| `small-kd` | CompactCNN Small | 175k | 686 KB | 686 KB | KD | TODO |

**Priority 2: Compression (same models)**

| Model ID | Architecture | Params | RAM | Storage | Precision | Status |
|----------|-------------|--------|-----|---------|-----------|--------|
| `tiny-bf16` | CompactCNN Tiny | 45k | 180 KB | 90 KB | BF16 | TODO |
| `small-bf16` | CompactCNN Small | 175k | 686 KB | 343 KB | BF16 | TODO |
| `small-fp16` | CompactCNN Small | 175k | 686 KB | 366 KB | FP16 | TODO |
| `small-int8` | CompactCNN Small | 175k | 686 KB | 175 KB | INT8 | TODO |

**Priority 3: Headphone Targets (1-2 MB RAM)**

| Model ID | Architecture | Params | RAM | Storage | Training | Status |
|----------|-------------|--------|-----|---------|----------|--------|
| `medium-direct` | CompactCNN Medium | 500k | 1.9 MB | 1.9 MB | Direct | TODO |
| `medium-kd` | CompactCNN Medium | 500k | 1.9 MB | 1.9 MB | KD | TODO |

**Reference: Over Budget (>2 MB)**

| Model ID | Architecture | Params | RAM | Notes |
|----------|-------------|--------|-----|-------|
| `cnn-rnn-direct` | CNN-RNN | 546k | 2.1 MB | Temporal modelling baseline |
| `large-direct` | CompactCNN Large | 1.2M | 4.7 MB | Upper accuracy bound only |

### Completed Experiments

_Results will be added here after training on YODAS-Granary._

---

## Compression Experiments

After training the best architecture, we evaluate quantisation for storage reduction:

| Precision | Storage Reduction | RAM Impact | Notes |
|-----------|------------------|------------|-------|
| FP32 | 1.0× (baseline) | None | Default |
| BF16 | 2.0× | None (CPU) | Near-lossless |
| FP16 | 2.0× | 39% savings | Web-compatible |
| INT8 | 4.0× | None (dequant.) | Requires dequantisation |
| INT4 | 8.0× | None (dequant.) | Needs QAT |

**Note:** Quantisation reduces storage (flash/SSD) but not runtime RAM unless native low-precision compute is available. On CPU (our target), low-precision weights are dequantised to FP32 at inference time.

---

## Training Protocol

### Standard Settings

```bash
# Train CompactCNN Small on 23 languages
uv run src/scripts/train_cnn.py \
  --dataset saattrupdan/yodas-granary-language-detection \
  --use-hf \
  --model-size small \
  --epochs 60 \
  --batch-size 32 \
  --lr 1e-4 \
  --weight-decay 1e-5
```

### Class Weighting

The dataset has significant class imbalance (Dutch: 13.5k samples, Latvian: 200 samples). We use inverse frequency weighting:

```python
weight_c = total_samples / (num_classes × count_per_class)
```

Enabled by default for N > 2 classes.

### Knowledge Distillation (Optional)

Train a small student using a larger teacher's soft targets:

```bash
uv run src/scripts/train_cnn.py \
  --kd true \
  --kd-teacher-checkpoint path/to/teacher.pth \
  --kd-alpha 0.7 \
  --kd-temperature 2.5
```

---

## Evaluation Metrics

### Primary

- **Overall accuracy** — Fraction of correct predictions
- **Per-language accuracy** — Accuracy for each individual language
- **Runtime RAM** — Model weights + activations + buffers (batch=1)

### Secondary

- **Storage size** — Compressed model size (flash/SSD)
- **FLOPs** — Computational complexity (not critical for 16kHz audio)
- **Latency** — Inference time on target hardware

---

## Scripts

| Script | Purpose |
|--------|---------|
| `train_cnn.py` | Train CompactCNN (Tiny/Small/Medium) on N languages |
| `train_cnn_rnn.py` | Train CNN-RNN on N languages |
| `compress_model.py` | Quantise to BF16/FP16/INT8 |
| `demo_server.py` | Flask server for web demo |
| `plot_pareto.py` | Generate Pareto frontier plot from results |

---

## Usage Examples

### Train a Model

```bash
uv run src/scripts/train_cnn.py \
  --dataset saattrupdan/yodas-granary-language-detection \
  --use-hf \
  --model-size small \
  --epochs 60
```

### Evaluate a Model

```bash
uv run src/scripts/evaluate.py \
  --checkpoint data/experiments/phase4b-small-23/model_best.pth \
  --dataset saattrupdan/yodas-granary-language-detection \
  --use-hf
```

### Generate Pareto Plot

```bash
uv run src/scripts/plot_pareto.py --output results/pareto_frontier.png
```

### Run Demo

```bash
uv run src/scripts/demo_server.py
# Open http://localhost:7860
```

---

## Target Devices

| Device | RAM Budget | Priority |
|--------|-----------|----------|
| Earbuds | <1 MB | Primary |
| Headphones | 1-2 MB | Secondary |

Models exceeding 2 MB RAM are not considered for deployment.

---

## Previous Work (Archive)

Historical experiments on Common Voice (binary Danish/English) are documented in the git history but not included here. Key findings:

- Wavelet features underperformed MFCC by 21 pp
- Compact CNN (Phase 4b) beat CNN-RNN by 5.9 pp on Common Voice
- Knowledge distillation added +0.98 pp on binary task

**This work focuses on multiclass (23 languages) on YODAS-Granary.**

---

**Training Order:**

1. **小 direct** — Baseline for earbud target (686 KB)
2. **小 KD** — Knowledge distillation (teacher: medium-direct)
3. **Tiny direct** — Ultra-compact baseline (180 KB)
4. **Medium direct** — Upper bound for headphones (1.9 MB)
5. **CNN-RNN direct** — Temporal modelling reference (2.1 MB)
6. **Compression** — BF16/FP16/INT8 on best Small model

After each training run, update `data/experiments/results.jsonl` with accuracy and regenerate the Pareto plot.
