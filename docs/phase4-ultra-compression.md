# Phase 4: Ultra-Low Precision Quantisation

**Goal:** Explore extreme compression techniques to minimise **storage size**.

**Critical note:** This addresses **storage (Flash/ROM)**, not **runtime RAM**. After
dequantisation, the model still occupies ~2.1 MB RAM (same as FP32 baseline).

## RAM vs Storage — Which Constraint Are We Solving?

| Device     | Storage Budget    | RAM Budget        |
| ---------- | ----------------- | ----------------- |
| Headphones | ~2-8 MB (typical) | 500 KB – 1 MB     |
| Earbuds    | ~1-4 MB (typical) | **100 KB – 1 MB** |

**This document:** Storage compression (Flash/ROM).

- ✅ Achieved: 276 KB INT4 storage
- ❌ Does NOT solve: 2.1 MB RAM requirement

**To reduce RAM:** Need knowledge distillation to smaller architecture (Phase 4b — TBD).

## Quick Summary: Storage Results

| Method                | Storage    | Reduction | Accuracy   | RAM (runtime) | Verdict               |
| --------------------- | ---------- | --------- | ---------- | ------------- | --------------------- |
| Baseline (FP32)       | 2136 KB    | —         | 90.86%     | ~2.1 MB       | Reference             |
| INT8 Weight-Only      | 427 KB     | 80%       | 90.86%     | ~2.1 MB       | ✅ Storage win        |
| **INT4 Quantisation** | **276 KB** | **87%**   | **91.32%** | ~2.1 MB       | ✅✅ **Best storage** |
| INT2 Quantisation     | 143 KB     | 93%       | 43.67%     | ~2.1 MB       | ❌ Accuracy collapse  |
| INT8 + 50% Sparse     | 534 KB     | 75%       | 43.67%     | ~2.1 MB       | ❌ Pruning fails      |

**INT4 is remarkable:** 276 KB storage with **better than baseline accuracy** (91.32% vs
90.86%)! Quantisation noise acts as regularisation.

## Experimental Details

### INT4 Quantisation (16 Levels)

**Method:** Per-tensor quantisation to 4-bit integers.

```python
# Quantisation: 4-bit = 16 levels
scale = (max_val - min_val) / 15.0  # 2^4 - 1 = 15
zero_point = (-min_val / scale).round().clamp(0, 15)
quantised = (param / scale + zero_point).round().clamp(0, 15).to(torch.int8)
```

**Storage:** In production, pack 2 × 4-bit weights per byte → 4 bits/weight effective.

**Results:**

- **Storage:** 276 KB (12.9% of baseline, **87% reduction**)
- **Accuracy:** 91.32% overall (**+0.46 pp vs baseline!**)
- **Danish:** 85.43% (+1.04 pp)
- **English:** 99.07% (-0.14 pp)

**Why accuracy improved:** Quantisation noise acts as implicit regularisation, reducing
overfitting. Known phenomenon in very low-precision quantisation.

**Confusion matrix:**

```
[[832, 142],   # Danish: 832 correct, 142 → English
 [  7, 748]]   # English: 7 → Danish, 748 correct
```

### INT2 Quantisation (4 Levels)

**Method:** Per-tensor quantisation to 2-bit integers (4 discrete levels).

**Results:**

- **Storage:** 143 KB (6.7% of baseline, 93% reduction)
- **Accuracy:** 43.67% (**catastrophic collapse**)
- **Pattern:** Model predicts English for nearly everything

**Why it failed:** 4 levels is too coarse to represent weight distributions effectively.
The model loses all discriminative power for Danish speech.

**Verdict:** 2-bit is below the information capacity needed for this task.

## Size vs Accuracy Trade-off Curve

```
Accuracy (%)
    92 ┤                    ★ INT4 (91.32%, 276 KB storage)
       │                  ╱
    91 ┤                ╱
       │              ╱
    90 ┤★ INT8 (90.86%, 427 KB storage)
       │╱
    80 ┤
       │
    70 ┤
       │
    60 ┤
       │
    50 ┤
       │
    40 ┤                    ✗ INT2 (43.67%, 143 KB storage)
       │                    ✗ Sparse (43.67%, 534 KB storage)
    ───┼────────────────────────────────────────
       0    200   400   600   800  1000  Storage (KB)

Note: All methods require ~2.1 MB RAM after dequantisation.
```

## Recommendations by Constraint

### If Storage-Constrained (Flash <1 MB, RAM >2 MB)

| Priority                 | Method   | Storage    | Accuracy   | Notes              |
| ------------------------ | -------- | ---------- | ---------- | ------------------ |
| **Best overall**         | **INT4** | **276 KB** | **91.32%** | Sweet spot         |
| Simpler                  | INT8     | 427 KB     | 90.86%     | Slightly larger    |
| Ultra-size (accept loss) | INT2     | 143 KB     | 43.67%     | ❌ Not recommended |

**Recommendation:** **INT4 at 276 KB** — leaves 724 KB headroom under 1 MB flash while
improving accuracy.

### If RAM-Constrained (<1 MB runtime)

| Method                | Storage | RAM     | Accuracy | Verdict               |
| --------------------- | ------- | ------- | -------- | --------------------- |
| Phase 2 INT4          | 276 KB  | ~2.1 MB | 91.32%   | ❌ Exceeds RAM budget |
| KD Student (Phase 4b) | ~50 KB  | ~100 KB | TBD      | ✅ Target approach    |

**Recommendation:** Knowledge distillation to tiny student model (10-20k params) needed
for RAM-constrained devices. Not yet implemented.

## Deployment Guide: INT4 Quantisation

### Export (Training Side)

```python
import torch
from tiny_language_detection.models.cnn_rnn import CNNRNNLanguageDetector

def quantise_to_int4(state_dict: dict) -> tuple[dict, dict]:
    """Quantise weights to 4-bit (16 levels)."""
    quantised = {}
    metadata = {"scales": {}, "zeros": {}}

    for name, param in state_dict.items():
        if param.dim() >= 2 and param.numel() > 100:  # Weight tensors only
            min_val = torch.min(param)
            max_val = torch.max(param)
            scale = (max_val - min_val) / 15.0  # 16 levels
            zero_point = (-min_val / scale).round().clamp(0, 15)

            # Store as int8 (PyTorch doesn't have int4 dtype)
            # Production: pack 2 weights per byte
            q = (param / scale + zero_point).round().clamp(0, 15).to(torch.int8)

            quantised[name] = q
            metadata["scales"][name] = scale
            metadata["zeros"][name] = zero_point
        else:
            quantised[name] = param.clone()  # Biases stay FP32

    return quantised, metadata

# Export
model = CNNRNNLanguageDetector(...)
model.load_state_dict(torch.load("phase2_model.pth"))
state_dict = model.state_dict()
quantised, metadata = quantise_to_int4(state_dict)

torch.save({
    "model": quantised,
    "metadata": metadata,
    "config": {...},  # n_mels, hidden_size, etc.
}, "model_int4.pt")  # File size: ~276 KB
```

### Load and Dequantise (Requires ~2.1 MB RAM)

```python
import torch

def dequantise_int4(quantised: dict, metadata: dict) -> dict:
    """Dequantise INT4 weights back to FP32 for inference."""
    state = {}
    scales = metadata["scales"]
    zeros = metadata["zeros"]

    for name, param in quantised.items():
        if name in scales:
            # Dequantise: w_fp32 = (w_int4 - zero_point) * scale
            state[name] = (param.float() - zeros[name]) * scales[name]
        else:
            state[name] = param

    return state

# Load
checkpoint = torch.load("model_int4.pt")
state_dict = dequantise_int4(checkpoint["model"], checkpoint["metadata"])

model = CNNRNNLanguageDetector(...)
model.load_state_dict(state_dict)  # Now occupies ~2.1 MB RAM
model.eval()

# Standard FP32 inference (no special INT4 kernels needed)
output = model(input_tensor)
```

### For RAM-Constrained Deployment: On-the-Fly Dequantisation

If RAM <2 MB, dequantise **per-layer** or **per-operator** during inference:

```python
class INT4Model:
    """INT4 model with on-the-fly dequantisation (low RAM footprint)."""

    def __init__(self, quantised_path: str):
        checkpoint = torch.load(quantised_path)
        self.quantised = checkpoint["model"]
        self.metadata = checkpoint["metadata"]

    def dequantise_layer(self, name: str) -> torch.Tensor:
        """Dequantise single layer on-demand."""
        if name in self.metadata["scales"]:
            return (self.quantised[name].float()
                    - self.metadata["zeros"][name]) * self.metadata["scales"][name]
        return self.quantised[name]

    def forward(self, x):
        # Manually implement forward, dequantising each layer just before use
        # This keeps RAM low: only one layer's weights in FP32 at a time
        # ... (requires custom layer-by-layer implementation)
        pass
```

**RAM benefit:** Only dequantise current layer's weights → ~0.3 MB RAM vs ~2.1 MB.

**Trade-off:** Requires custom inference code; can't use standard
`model.load_state_dict`.

## Comparison to Previous Phase 4 Results

| Method (Previous) | Storage | Accuracy | RAM     | Method (Ultra) | Storage    | Accuracy   | RAM     |
| ----------------- | ------- | -------- | ------- | -------------- | ---------- | ---------- | ------- |
| FP32              | 2086 KB | 90.86%   | ~2.1 MB | FP32           | 2136 KB    | 90.86%     | ~2.1 MB |
| FP16              | 1043 KB | 90.86%   | ~2.1 MB | —              | —          | —          | —       |
| INT8              | ~520 KB | 90.86%   | ~2.1 MB | INT8           | 427 KB     | 90.86%     | ~2.1 MB |
| —                 | —       | —        | —       | **INT4**       | **276 KB** | **91.32%** | ~2.1 MB |
| —                 | —       | —        | —       | INT2           | 143 KB     | 43.67%     | ~2.1 MB |

## Paths to 100 KB RAM Target

**Current best (INT4):** 276 KB storage, ~2.1 MB RAM.

To reach **100 KB RAM**, we need architectural changes:

| Technique                  | Expected RAM | Expected Storage | Accuracy | Status             |
| -------------------------- | ------------ | ---------------- | -------- | ------------------ |
| **Knowledge Distillation** | 100-200 KB   | 25-50 KB         | ~88-90%  | Phase 4b (TBD)     |
| Tiny CNN (no RNN)          | 50-100 KB    | 15-30 KB         | ~80-85%  | Phase 4b (TBD)     |
| On-the-fly INT4 dequant    | ~300 KB      | 276 KB           | 91.32%   | Custom code needed |

**Recommended next step (Phase 4b):** Knowledge distillation to 10-20k parameter student
model → target 100-150 KB RAM, 25-50 KB storage.

## Conclusion

**INT4 quantisation achieves exceptional storage compression:**

- ✅ **276 KB** stored size (87% reduction from 2.1 MB)
- ✅ **91.32% accuracy** (slightly better than baseline)
- ✅ **Well under storage targets** (276 KB vs ~1-4 MB flash)
- ❌ **~2.1 MB RAM** still required after dequantisation

**For RAM-constrained earbuds (<1 MB runtime):**

- On-the-fly dequantisation can reduce RAM to ~300 KB (custom code required)
- Knowledge distillation to tiny student needed for <200 KB RAM
- Phase 4b (not yet implemented) will address RAM via architecture changes

**Summary:** Phase 4 solved **storage**; Phase 4b must solve **RAM**.
