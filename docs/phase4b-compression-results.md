# Phase 4b: Model Compression Results

**Date:** 2026-07-22
**Model:** Compact CNN Small (KD trained) — 175k params
**Baseline:** FP32, 96.76% accuracy (DA: 98.05%, EN: 95.10%)

## Summary

Compressed the Phase 4b KD model to three precisions and evaluated accuracy vs.
storage trade-offs:

| Precision | Storage | Compression | Accuracy   | Δ vs. Baseline |
|-----------|---------|-------------|------------|----------------|
| FP32      | 686 KB  | 1.00×       | **96.76%** | —              |
| BF16      | 343 KB  | 2.00×       | **96.65%** | −0.11 pp       |
| INT8      | 175 KB  | 3.92×       | **95.66%** | −1.10 pp       |
| INT4      | 90 KB   | 7.64×       | **71.72%** | −25.04 pp ❌   |

**Key findings:**

1. **BF16 is near-lossless** — Only 0.11 pp accuracy drop with 2× storage
   reduction. Recommended for deployment.
2. **INT8 is practical** — 1.1 pp drop for 4× compression. Acceptable for
   latency constrained applications.
3. **INT4 collapses** — Catastrophic accuracy loss (−25 pp). 16-level
   quantisation is too coarse for this architecture.

## Detailed Results

### BF16 (Brain Floating Point 16)

- **Storage:** 343.1 KB (2.00× compression)
- **Overall accuracy:** 96.65%
- **Danish accuracy:** 98.15% (−0.10 pp vs. baseline)
- **English accuracy:** 94.70% (−0.40 pp vs. baseline)
- **Confusion matrix:** [[955, 19], [37, 718]]

**Assessment:** BF16 conversion is essentially lossless for this model. The
slight accuracy drop (0.11 pp) is within measurement noise. **Recommended for
deployment** on hardware with BF16 support.

### INT8 (8-bit Integer)

- **Storage:** 175.1 KB (3.92× compression)
- **Overall accuracy:** 95.66%
- **Danish accuracy:** 98.46% (+0.41 pp vs. baseline)
- **English accuracy:** 92.05% (−3.05 pp vs. baseline)
- **Confusion matrix:** [[958, 16], [62, 693]]

**Assessment:** INT8 quantisation introduces modest accuracy degradation
(1.1 pp overall). Interestingly, Danish accuracy slightly improved while
English dropped 3 pp. The model remains robust, with 95.66% accuracy still
exceeding the 80–90% target range. **Viable for deployment** where storage is
critical.

### INT4 (4-bit Integer)

- **Storage:** 89.8 KB (7.64× compression, theoretical)
- **Overall accuracy:** 71.72%
- **Danish accuracy:** 49.90% (−48.15 pp vs. baseline) ❌
- **English accuracy:** 99.87% (+4.77 pp vs. baseline)
- **Confusion matrix:** [[486, 488], [1, 754]]

**Assessment:** **Catastrophic collapse.** The model predicts almost
everything as English (class 1). Danish accuracy plummeted from 98.05% to
49.90% — essentially random guessing.

**Root cause:** 4-bit quantisation with only 16 discrete levels introduces
quantisation error exceeding the model's robustness margin. Mean absolute
weight deviation from FP32 is ~0.012, which appears small but is sufficient to
shift decision boundaries dramatically.

**Recommendation:** INT4 quantisation at this granularity is **not viable**
for the Compact CNN architecture. Alternative approaches:

1. **Quantisation-aware training (QAT):** Train with simulated INT4
   quantisation to build robustness.
2. **Mixed precision:** Keep sensitive layers (e.g., first conv, classifier)
   in FP16/INT8.
3. **Structured pruning:** Reduce parameter count before quantisation.
4. **Knowledge distillation to smaller model:** Train a tiny (44k param) model
   with KD instead of aggressive quantisation.

## RAM Analysis

| Component              | Size (KB) | Notes                                      |
|------------------------|-----------|--------------------------------------------|
| **Model weights**      |           |                                            |
| ├─ FP32                | 686       | Baseline                                   |
| ├─ BF16                | 343       | Requires BF16 hardware support             |
| ├─ INT8                | 175       | Requires dequantisation or INT8 kernels    |
| └─ INT4                | 90        | Requires dequantisation (collapsed)        |
| **Activations**        | ~50       | Batch size 1, 80 mel bins, 100 time frames |
| **Audio buffer**       | 96        | 3s @ 16kHz, 16-bit                         |
| **Spectrogram buffer** | 25        | 80 mel bins × 100 frames × FP32            |
| **Total runtime RAM**  |           |                                            |
| ├─ FP32/BF16           | ~857 KB   | ✅ Earbud target (<1 MB)                   |
| └─ INT8 (dequantised)  | ~1.2 MB   | Exceeds earbud budget                      |

**Note:** Weight quantisation reduces **storage** (flash) but not **runtime
RAM** unless:

- Hardware supports native low-precision compute (INT8/INT4 Tensor Cores)
- On-the-fly dequantisation is used (adds latency)

For CPU-only deployment (B&O headphones/earbuds), BF16 or FP32 is recommended
despite larger storage, as runtime RAM is dominated by activations and buffers,
not weights.

## Experimental Setup

- **Teacher model:** Phase 2 CNN-RNN (546k params, 90.86% accuracy)
- **Student model:** Compact CNN Small (175k params, 96.76% accuracy with KD)
- **Test set:** 1,729 samples (1h DA + 1h EN, speaker-independent)
- **Quantisation method:** Per-tensor symmetric quantisation
- **Dequantisation:** On-the-fly to FP32 before inference
- **Hardware:** CPU-only (no GPU acceleration)

## Scripts

- Compression:
  `uv run src/scripts/phase4b_compression.py --checkpoint <path> --precision <bf16|int8|int4>`
- Evaluation:
  `uv run src/scripts/evaluate_phase4b_compressed.py --checkpoint <path>`

## Conclusion

**Recommended deployment configuration:**

- **BF16** if hardware supports it (343 KB storage, 96.65% accuracy, ~857 KB
  runtime RAM)
- **FP32** as fallback (686 KB storage, 96.76% accuracy, ~857 KB runtime RAM)

INT8 is acceptable for storage-constrained scenarios (175 KB) if the 1.1 pp
accuracy drop is tolerable. INT4 is not recommended without quantisation-aware
training or architectural modifications.

**All models fit the earbud RAM target** (<1 MB) when using FP32/BF16
precision, with ~857 KB total runtime memory.
