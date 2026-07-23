# Experiments: Multilingual Language Detection

**Project:** Tiny Language Detection  
**Repository:** https://github.com/alexandrainst/tiny_language_detection  
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

Results are tracked in `data/experiments/results.jsonl`. To visualise:

```bash
uv run src/scripts/plot_pareto.py --highlight phase4b-small-23
```

This generates `results/pareto_frontier.png` showing all experiments with the Pareto frontier.

### Pending Experiments

| Model ID | Architecture | Params | RAM (KB) | Status |
|----------|-------------|--------|----------|--------|
| `phase4b-tiny-23` | CompactCNN (Tiny) | 45k | 180 | TODO: Train |
| `phase4b-small-23` | CompactCNN (Small) | 175k | 686 | TODO: Train |
| `phase4b-medium-23` | CompactCNN (Medium) | 500k | 1950 | TODO: Train |
| `phase2-cnn-rnn-23` | CNN-RNN | 546k | 2100 | TODO: Train |

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

**Next Steps:**
1. Train baseline CompactCNN models (Tiny/Small/Medium) on 23 languages
2. Evaluate per-language accuracy and identify weak languages
3. Generate Pareto frontier plot
4. Select best model for target RAM budget
5. Apply compression (BF16/INT8) for storage reduction
