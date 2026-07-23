# Phase 4b: Compact CNN Results

**Training completed:** 2026-07-22
**Refactored for multi-class:** 2026-07-23

## Executive Summary

A compact CNN (175k params, ~686 KB RAM) achieves **96.76% accuracy** — surpassing Phase
2 (90.86%, 2.1 MB) by **5.9 percentage points** while using only **33% of the RAM**.

**Recommended for deployment:** Small model with Knowledge Distillation.

**Note:** The original Phase 4b experiments used `src/scripts/train_phase4b.py` (binary
DA/EN only). The architecture has been refactored into
`src/scripts/train_multiclass_cnn.py` for flexible N-class training with the same model
class (`CompactCNNLanguageDetector`).

## Final Results

| Model                     | Params   | RAM (FP32)  | Best Acc   | Danish | English | Training Mode     |
| ------------------------- | -------- | ----------- | ---------- | ------ | ------- | ----------------- |
| Phase 1 (MFCC+CNN)        | 56k      | ~224 KB     | 81.7%      | —      | —       | Direct            |
| Phase 2 (Log-Mel+CNN-RNN) | 546k     | ~2.1 MB     | 90.86%     | 84.39% | 99.21%  | Direct            |
| Phase 3 (Wavelet+CNN)     | 56k      | ~224 KB     | 60.67%     | 43.63% | 82.65%  | Direct            |
| **Phase 4b Small**        | **175k** | **~686 KB** | **95.78%** | 92.09% | 97.75%  | Direct            |
| **Phase 4b Small (KD)**   | **175k** | **~686 KB** | **96.76%** | 93.33% | 97.88%  | KD (α=0.7, T=2.5) |

### Key Findings

1. **Direct training achieved 95.78%** — already beating Phase 2 by 4.9 pp
2. **Knowledge distillation added +0.98 pp** for 96.76% total
3. **Danish accuracy improved dramatically:** 93.33% vs 84.39% (Phase 2)
4. **English accuracy maintained:** 97.88% vs 99.21% (Phase 2)
5. **Model fits earbud RAM budget:** 686 KB < 1 MB target

## Training Dynamics

### Direct Training

- **Rapid convergence:** 84% by epoch 8, 95% by epoch 25
- **Final best:** Epoch 31 at 95.78%
- **Stable:** Mild oscillation, no severe overfitting
- **Training acc:** 97.87%, **Test acc:** 95.78% (Δ = 2.1 pp)

### Knowledge Distillation

- **Slower start:** High KD loss initially, 68% at epoch 7
- **Strong convergence:** Epoch 20-37 rapid improvement
- **Final best:** Epoch 37 at 96.76%
- **Training acc:** 98.30%, **Test acc:** 96.76% (Δ = 1.5 pp)
- **Smoother learning:** KD avoided noise spikes seen in direct training

## Confusion Matrix (KD Model) — Training Log

```text
              Predicted
              DA    EN
Actual DA  [  909   65 ]  → DA precision: 93.3%
Actual EN  [   16  739 ]  → EN precision: 97.9%
```text

- **Danish:** 909/974 correct (93.33%)
- **English:** 739/755 correct (97.88%)
- **Total:** 1,648/1,729 correct (95.32% final epoch, 96.76% best epoch)

## Verified Evaluation Results

Independent evaluation of `model_best.pth` confirms **96.76% overall**:

```text
              Predicted
              DA    EN
Actual DA  [  955   19 ]  → DA: 98.05%
Actual EN  [   37  718 ]  → EN: 95.10%
```text

- **Overall:** 96.76% (1,673/1,729 correct)
- **Danish:** 98.05% (955/974)
- **English:** 95.10% (718/755)

**Notable:** Phase 4b reverses Phase 2's weakness — **Danish now outperforms English**
(98.05% vs 95.10%), versus Phase 2's Danish 84.39% vs English 99.21%.

## Accuracy by Duration (KD Model)

| Duration | Accuracy | Samples |
| -------- | -------- | ------- |
| 0-2s     | 94.30%   | ~432    |
| 2-4s     | 95.91%   | ~432    |
| 4-6s     | 95.85%   | ~432    |
| 6+s      | 93.31%   | ~433    |

**Observation:** All duration groups perform well (93-96%), no major degradation on long
clips — suggests robust temporal modelling via global pooling.

## Model Architecture (Small)

**Original Phase 4b (binary DA/EN):**

```text
Input: [batch, 1, 80, time]

Block 1: Conv2d(1→32) + BN + ReLU + MaxPool(2,2)  # 80→40
Block 2: Conv2d(32→64) + BN + ReLU + MaxPool(2,2) # 40→20
Block 3: Conv2d(64→128) + BN + ReLU + MaxPool(2,2) # 20→10

Global Pool: Mean(time) → [batch, 1280]
Classifier: Linear(1280→64) + ReLU + Dropout(0.3) + Linear(64→2)

Total: 175,234 parameters
```text

**Multi-class refactoring (`train_multiclass_cnn.py`):**

The same architecture is now configurable via CLI:

```bash
uv run src/scripts/train_multiclass_cnn.py \
  --channels 32 64 128 \
  --hidden-size 128 \
  --dropout 0.3 \
  --num-languages 23  # Auto-detected if omitted
```text

For comparison with CNN-RNN, use `src/scripts/train_multiclass_cnn_rnn.py` with
equivalent `--channels` and `--hidden-size` arguments.

## Comparison with Phase 2

### Why Phase 4b Beats Phase 2

| Aspect       | Phase 2   | Phase 4b Small (Verified) |
| ------------ | --------- | ------------------------- |
| Architecture | CNN + GRU | CNN only                  |
| Parameters   | 546k      | 175k (3.1× smaller)       |
| RAM          | ~2.1 MB   | ~686 KB (3.1× smaller)    |
| Danish acc   | 84.39%    | **98.05% (+13.66 pp)** ✅ |
| English acc  | 99.21%    | 95.10% (-4.11 pp)         |
| Overall      | 90.86%    | **96.76% (+5.90 pp)** ✅  |

**Hypothesis:** The GRU in Phase 2 was overkill for binary classification. The simpler
CNN with global pooling captures sufficient temporal patterns while being easier to
optimise. Phase 4b's larger convolutional filters (despite fewer total params) have more
capacity for spectro-temporal features.

### What Phase 2 Did Better

- English accuracy: 99.21% vs 97.88% (Δ = -1.33 pp)
- For applications where **English-only** accuracy is critical, Phase 2 INT8 compressed
  may still be preferable

## Deployment Readiness

### Storage Size (INT4 Quantisation)

Expected INT4 storage: ~220 KB (calculated: 175k × 0.5 bytes/param)

```bash
# Export (add to phase4b_compression.py script)
python src/scripts/phase4b_compression.py \
  --checkpoint data/experiments/phase4b/tiny_cnn_kd/model_best.pth \
  --output data/experiments/phase4b/tiny_cnn_kd_int4.pt \
  --quantise int4
```text

### RAM Budget Breakdown

| Component                         | Size           |
| --------------------------------- | -------------- |
| Model weights (FP32)              | ~686 KB        |
| Activations (batch=1)             | ~50 KB         |
| Audio buffer (16-bit, 3s @ 16kHz) | ~96 KB         |
| Mel spectrogram buffer            | ~25 KB         |
| **Total**                         | **~857 KB** ✅ |

**Fits within:**

- Earbuds (<1 MB): ✅
- Headphones (2-3 MB): ✅

## Model Compression Results

Compressed the KD model to three precisions (see `docs/phase4b-compression-results.md`):

| Precision | Storage | Compression | Accuracy   | Δ vs. Baseline |
|-----------|---------|-------------|------------|----------------|
| FP32      | 686 KB  | 1.00×       | **96.76%** | —              |
| BF16      | 343 KB  | 2.00×       | **96.65%** | −0.11 pp       |
| INT8      | 175 KB  | 3.92×       | **95.66%** | −1.10 pp       |
| INT4      | 90 KB   | 7.64×       | **71.72%** | −25.04 pp ❌   |

**Recommendations:**

- **BF16** for deployment if hardware supports it (near-lossless, 343 KB)
- **INT8** acceptable if storage critical (1.1 pp drop, 175 KB)
- **INT4 not viable** without quantisation-aware training (catastrophic collapse)

## Files Created

**Original Phase 4b (binary DA/EN):**

- `src/tiny_language_detection/models/tiny_cnn.py` — Model architecture
- `src/scripts/train_phase4b.py` — Training script (direct + KD modes)
- `src/scripts/phase4b_compression.py` — Compression script (BF16/INT8/INT4)
- `src/scripts/evaluate_phase4b_compressed.py` — Evaluation script
- `docs/phase4b-compact-cnn.md` — Training guide and deployment instructions
- `docs/phase4b-compression-results.md` — Compression experiment results
- `data/experiments/phase4b/tiny_cnn_direct/` — Direct training outputs
- `data/experiments/phase4b/tiny_cnn_kd/` — KD training outputs (recommended)

**Multi-class refactoring (2026-07-23):**

- `src/tiny_language_detection/data/multiclass_dataset.py` — Shared dataset class
- `src/tiny_language_detection/training.py` — Shared training utilities
- `src/scripts/train_multiclass_cnn.py` — CNN training (N classes)
- `src/scripts/train_multiclass_cnn_rnn.py` — CNN-RNN training (N classes)
- `src/tiny_language_detection/models/cnn_rnn.py` — Updated with configurable channels

## Next Steps

**Completed (2026-07-23):**

1. ✅ **Multi-class extension** — Refactored into `train_multiclass_cnn.py` and
   `train_multiclass_cnn_rnn.py` supporting N classes
2. ✅ **Architecture comparison** — CNN vs CNN-RNN scripts with configurable parameters

**Remaining:**

1. **Train on 23 languages** — Run `train_multiclass_cnn.py` on YODAS-Granary dataset
2. **INT8 export** — Quantise best model for storage optimisation (~175 KB)
3. **Hardware testing** — Deploy to B&O target device, measure real-world latency
4. **Ablation study** — Test tiny (44k) and medium (422k) variants for accuracy/RAM
   trade-off
5. **Quantisation-aware training** — If INT4 storage needed, train with simulated
   low-precision to build robustness

## Reproduction Commands

### Train Small Model (Direct)

**Original script (binary DA/EN only):**

```bash
uv run src/scripts/train_phase4b.py \
  --model-size small \
  --mode direct \
  --epochs 50 \
  --batch-size 64 \
  --lr 0.001
```text

**New multi-class script (recommended):**

```bash
uv run src/scripts/train_multiclass_cnn.py \
  --dataset saattrupdan/yodas-granary-language-detection \
  --use-hf \
  --channels 32 64 128 \
  --hidden-size 128 \
  --dropout 0.3 \
  --epochs 60 \
  --batch-size 32 \
  --lr 1e-4 \
  --weight-decay 1e-4
```text

### Train Small Model (Knowledge Distillation)

Knowledge distillation is not yet implemented in the multi-class script. Use the original
Phase 4b script for KD training (binary only).

### Evaluate Best Model

```bash
uv run src/scripts/evaluate_phase4b.py \
  --checkpoint data/experiments/phase4b/tiny_cnn_kd/model_best.pth
```text

### Multi-Class Training (23 Languages)

For training on the YODAS-Granary 23-language dataset:

```bash
# Compact CNN (recommended baseline)
uv run src/scripts/train_multiclass_cnn.py \
  --dataset saattrupdan/yodas-granary-language-detection \
  --use-hf \
  --channels 32 64 128 \
  --hidden-size 128 \
  --epochs 60

# CNN-RNN (for comparison)
uv run src/scripts/train_multiclass_cnn_rnn.py \
  --dataset saattrupdan/yodas-granary-language-detection \
  --use-hf \
  --channels 32 64 128 \
  --hidden-size 64 \
  --num-layers 1 \
  --epochs 60
```text

## Conclusion

Phase 4b's compact CNN with knowledge distillation achieves **96.76% accuracy** with
**686 KB RAM** — the best accuracy-to-size ratio of all phases. The model is ready for
deployment testing on B&O hardware and provides a strong foundation for multi-class
extension.
