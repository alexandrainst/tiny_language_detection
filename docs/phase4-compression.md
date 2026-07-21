# Phase 4: Model Compression and Optimisation

**Goal:** Compress the Phase 2 (Log-Mel + CNN-RNN) model to meet edge device memory
constraints while maintaining accuracy.

**Target Platform:** CPU-only (B&O headphones and earbuds)

## Critical Distinction: Storage vs RAM

| Aspect                   | What It Means                  | Our Constraint         |
| ------------------------ | ------------------------------ | ---------------------- |
| **Storage (Flash/ROM)**  | Model file size on disk        | Less critical          |
| **RAM (Runtime Memory)** | Memory needed during inference | **Primary constraint** |

**Weight quantisation (INT8/INT4) reduces storage but NOT RAM** (unless you have native
low-precision compute). When you dequantise weights at load time, they expand back to
FP32 in RAM.

## RAM Targets (Runtime Memory)

| Device     | RAM Budget    |
| ---------- | ------------- |
| Headphones | 500 KB – 1 MB |
| Earbuds    | 100 KB – 1 MB |

**Current Phase 2 model RAM usage:**

- Weights (FP32): ~2.1 MB
- Activations + overhead: ~0.5 MB
- **Total: ~2.6 MB** ❌ Exceeds both targets

## Compression Techniques Tested

### 1. Weight Quantisation (INT8/INT4) — Storage Only

| Method            | Storage | RAM (after dequantise) | Accuracy | Verdict        |
| ----------------- | ------- | ---------------------- | -------- | -------------- |
| FP32 Baseline     | 2.09 MB | ~2.1 MB                | 90.86%   | Reference      |
| INT8 Weight-Only  | 0.52 MB | ~2.1 MB                | 90.86%   | ✅ Storage win |
| INT4 Quantisation | 0.28 MB | ~2.1 MB                | 91.32%   | ✅ Storage win |

**Good for:** Storage-constrained devices with adequate RAM.

**Not sufficient for:** RAM-constrained earbuds (<1 MB runtime budget).

### 2. Magnitude Pruning — No Storage/RAM Benefit

| Pruning | Storage\* | RAM\*   | Accuracy | Verdict          |
| ------- | --------- | ------- | -------- | ---------------- |
| 20%     | 1.04 MB   | ~2.1 MB | 88.66%   | ❌ Accuracy loss |
| 40%+    | 1.04 MB   | ~2.1 MB | 44%      | ❌ Collapse      |

\* Unstructured pruning zeros weights but doesn't reduce storage/RAM without sparse
tensor support.

### 3. What Would Actually Reduce RAM

To meet the **100 KB – 1 MB RAM** target, we need:

| Technique                  | How It Works                                          | Potential      |
| -------------------------- | ----------------------------------------------------- | -------------- |
| **Knowledge Distillation** | Train smaller student model using Phase 2 as teacher  | 50-200 KB RAM  |
| **Architecture Redesign**  | Build tiny model from scratch (fewer params)          | 50-200 KB RAM  |
| **Structured Pruning**     | Remove entire filters/neurons (not just zero weights) | 200-500 KB RAM |
| **Native INT4 Inference**  | On-the-fly dequantisation + INT4 compute              | ~0.3 MB RAM    |

**Current status:** Not yet implemented. Phase 4 focused on quantisation which solves
storage, not RAM.

## Summary: Storage Achieved, RAM Not Yet Addressed

| Metric                   | Target        | Achieved?             | Notes                |
| ------------------------ | ------------- | --------------------- | -------------------- |
| **Storage (ears <1 MB)** | <1 MB         | ✅ **0.28 MB (INT4)** | 72% under target!    |
| **RAM (ears <1 MB)**     | <1 MB         | ❌ **~2.1 MB**        | 2.1× over budget     |
| **RAM (headphones)**     | 500 KB – 1 MB | ❌ **~2.1 MB**        | 2.1–4.2× over budget |
| **Accuracy**             | >90%          | ✅ **91.32% (INT4)**  | Exceeds baseline     |

## Recommendation: Next Steps for RAM Reduction

### Phase 4b: Knowledge Distillation (Recommended)

Train a compact student model:

- **Target:** 10-20k parameters (vs 546k current)
- **Architecture:** Single CNN + global pooling (no RNN/GRU)
- **Method:** Distill from Phase 2 INT4 teacher
- **Expected RAM:** ~50-100 KB (weights) + ~50 KB (activations) = **100-150 KB total**
- **Expected storage:** ~25-50 KB (INT4 packed)

### Alternative: Native Low-Precision Inference

If target hardware supports INT4/INT8 compute:

- Keep Phase 2 INT4 model
- Implement on-the-fly dequantisation (per-weight, not all at once)
- **RAM:** ~0.3 MB (only need buffers for current layer's dequantised weights)
- **Challenge:** Requires custom inference code per platform

## Deployment Guide: INT4 for Storage-Constrained Devices

If your device has **adequate RAM (>2 MB)** but limited flash:

### Export (Training Side)

```python
def quantise_to_int4(state_dict: dict) -> tuple[dict, dict]:
    """Quantise weights to 4-bit (16 levels)."""
    quantised = {}
    metadata = {"scales": {}, "zeros": {}}

    for name, param in state_dict.items():
        if param.dim() >= 2 and param.numel() > 100:
            min_val = torch.min(param)
            max_val = torch.max(param)
            scale = (max_val - min_val) / 15.0  # 16 levels
            zero_point = (-min_val / scale).round().clamp(0, 15)
            q = (param / scale + zero_point).round().clamp(0, 15).to(torch.int8)
            quantised[name] = q
            metadata["scales"][name] = scale
            metadata["zeros"][name] = zero_point
        else:
            quantised[name] = param.clone()

    return quantised, metadata
```

### Load and Dequantise (Requires >2 MB RAM)

```python
def dequantise_int4(quantised: dict, metadata: dict) -> dict:
    """Dequantise INT4 weights to FP32 for inference."""
    state = {}
    for name, param in quantised.items():
        if name in metadata["scales"]:
            state[name] = (param.float() - metadata["zeros"][name]) * metadata["scales"][name]
        else:
            state[name] = param
    return state

# Load
checkpoint = torch.load("model_int4.pt")
state_dict = dequantise_int4(checkpoint["model"], checkpoint["metadata"])
model.load_state_dict(state_dict)  # Now occupies ~2.1 MB RAM
```

## Conclusion

**Phase 4 achieved storage compression but not RAM reduction:**

- ✅ **Storage:** 0.28 MB (INT4) — well under <1 MB flash target
- ✅ **Accuracy:** 91.32% — best across all phases
- ❌ **RAM:** ~2.1 MB — exceeds both earbud (100 KB-1 MB) and headphone (500 KB-1 MB)
  targets

**Next step:** Knowledge distillation to a tiny student model (10-20k params) to achieve
<200 KB RAM runtime. This is Phase 4b — not yet implemented.
