# Multi-Class Training Guide

**Created:** 2026-07-23

Training scripts for N-language classification (2–23+ languages) using Compact CNN and
CNN-RNN architectures.

## Scripts

| Script                              | Model      | Best For                       |
| ----------------------------------- | ---------- | ------------------------------ |
| `train_multiclass_cnn.py`           | CNN only   | **Recommended baseline**       |
| `train_multiclass_cnn_rnn.py`       | CNN + GRU  | If temporal modelling critical |

Both scripts share the same data loading, feature extraction, and training loop — only
the model architecture differs.

## Quick Start

### Binary Classification (Danish vs English)

```bash
# Compact CNN (recommended)
uv run src/scripts/train_multiclass_cnn.py \
  --dataset saattrupdan/yodas-granary-language-detection \
  --use-hf \
  --epochs 60
```text

### 23-Language Classification (YODAS-Granary)

```bash
# Compact CNN
uv run src/scripts/train_multiclass_cnn.py \
  --dataset saattrupdan/yodas-granary-language-detection \
  --use-hf \
  --epochs 60 \
  --batch-size 32 \
  --lr 1e-4 \
  --weight-decay 1e-4

# CNN-RNN (for comparison)
uv run src/scripts/train_multiclass_cnn_rnn.py \
  --dataset saattrupdan/yodas-granary-language-detection \
  --use-hf \
  --epochs 60 \
  --hidden-size 64 \
  --num-layers 1
```text

### Local Dataset (CSV Manifest)

```bash
uv run src/scripts/train_multiclass_cnn.py \
  --dataset data/manifest.csv \
  --data-dir data/my-dataset \
  --epochs 60
```text

**Expected directory structure:**

```text
data/my-dataset/
  da/
    audio001.wav
    audio002.wav
  en/
    audio003.wav
  sv/
    audio004.wav
```text

## Architecture Parameters

### Compact CNN

| Argument          | Default      | Description                  |
| ----------------- | ------------ | ---------------------------- |
| `--channels`      | `32 64 128`  | CNN channels per block       |
| `--hidden-size`   | `128`        | Classifier hidden size       |
| `--dropout`       | `0.3`        | Dropout probability          |

**Example: Small model (~100k params)**

```bash
uv run src/scripts/train_multiclass_cnn.py \
  --dataset ... --use-hf \
  --channels 16 32 64 \
  --hidden-size 64
```text

**Example: Large model (~500k params)**

```bash
uv run src/scripts/train_multiclass_cnn.py \
  --dataset ... --use-hf \
  --channels 64 128 256 \
  --hidden-size 256
```text

### CNN-RNN

| Argument          | Default      | Description                  |
| ----------------- | ------------ | ---------------------------- |
| `--channels`      | `32 64 128`  | CNN channels per block       |
| `--hidden-size`   | `64`         | GRU hidden dimension         |
| `--num-layers`    | `1`          | Number of GRU layers         |
| `--dropout`       | `0.3`        | Dropout probability          |

**Example: 2-layer GRU**

```bash
uv run src/scripts/train_multiclass_cnn_rnn.py \
  --dataset ... --use-hf \
  --hidden-size 128 \
  --num-layers 2
```text

## Feature Extraction

Both scripts use Mel spectrograms with configurable parameters:

| Argument       | Default | Description              |
| -------------- | ------- | ------------------------ |
| `--n-mels`     | `80`    | Number of Mel bins       |
| `--n-fft`      | `400`   | FFT window size          |
| `--hop-length` | `160`   | Hop length (10ms @16kHz) |
| `--f-min`      | `0.0`   | Minimum frequency (Hz)   |
| `--f-max`      | `None`  | Maximum frequency (Hz)   |

**Example: Smaller features for edge deployment**

```bash
uv run src/scripts/train_multiclass_cnn.py \
  --dataset ... --use-hf \
  --n-mels 40 \
  --n-fft 256
```text

## Data Augmentation (SpecAugment)

| Argument       | Default | Description                |
| -------------- | ------- | -------------------------- |
| `--time-mask`  | `3`     | Time mask length           |
| `--freq-mask`  | `2`     | Frequency mask length      |
| `--time-masks` | `1`     | Number of time masks       |
| `--freq-masks` | `1`     | Number of frequency masks  |

**Example: Strong augmentation**

```bash
uv run src/scripts/train_multiclass_cnn.py \
  --dataset ... --use-hf \
  --time-masks 3 \
  --freq-masks 2
```text

## Training Hyperparameters

| Argument          | Default | Description           |
| ----------------- | ------- | --------------------- |
| `--epochs`        | `60`    | Training epochs       |
| `--batch-size`    | `32`    | Batch size            |
| `--lr`            | `1e-4`  | Learning rate         |
| `--weight-decay`  | `1e-4`  | L2 regularisation     |
| `--max-grad-norm` | `0.5`   | Gradient clipping     |

## Class Weighting

Enabled by default for N > 2 classes to handle imbalance (e.g., Dutch 13.5k vs Latvian
200 in YODAS-Granary).

```bash
# Enable (default)
uv run src/scripts/train_multiclass_cnn.py --dataset ... --use-hf

# Disable
uv run src/scripts/train_multiclass_cnn.py --dataset ... --use-hf --no-class-weights
```text

## Output

Checkpoints and logs are saved to `data/experiments/{dataset_name}/`:

```text
data/experiments/yodas-granary-language-detection/
  checkpoint_epoch_01.pth.tar
  checkpoint_epoch_02.pth.tar
  ...
  best_model.pth
  config.json
```text

## Monitoring Training

Training logs show per-language accuracy. For N > 5, shows top 5 and bottom 5 languages:

```text
Epoch 25/60 | LR: 2.1e-05 | Train: 97.23% | Test: 89.45% | \
  Top: da:94%, en:96%, sv:92%, no:91%, de:90%, \
  Bottom: lv:78%, lt:80%, el:82%, et:84%, hu:85%
```text

## Reproduction Commands

### Phase 4b Equivalent (Binary DA/EN)

```bash
uv run src/scripts/train_multiclass_cnn.py \
  --dataset saattrupdan/yodas-granary-language-detection \
  --use-hf \
  --channels 32 64 128 \
  --hidden-size 128 \
  --dropout 0.3 \
  --epochs 60
```text

### CNN-RNN Baseline

```bash
uv run src/scripts/train_multiclass_cnn_rnn.py \
  --dataset saattrupdan/yodas-granary-language-detection \
  --use-hf \
  --channels 32 64 128 \
  --hidden-size 64 \
  --num-layers 1 \
  --epochs 60
```text

## Tips

1. **Start with CNN** — Phase 4b showed CNN outperforms CNN-RNN for binary classification
2. **Monitor per-language accuracy** — Identify weak languages early
3. **Class weights matter** — Keep enabled for N > 2 to avoid majority-class bias
4. **60 epochs sufficient** — Convergence typically happens by epoch 40-50
5. **LR 1e-4 + weight decay** — More stable than Phase 2's 1e-3 Adam without decay

## See Also

- `docs/phase4b-results.md` — Phase 4b binary classification results
- `docs/granary-dataset.md` — YODAS-Granary dataset documentation
- `docs/phase4b-compression-results.md` — Model compression experiments
