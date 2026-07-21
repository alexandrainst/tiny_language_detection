# Phase 4b: Compact CNN Models for Low-RAM Deployment

**Goal:** Train compact CNN models that fit within RAM budgets (100 KB – 1 MB) while
achieving 80-90% accuracy for Danish vs English language detection.

**Key insight:** The original tiny model (~5k params) was too small, achieving only ~65%
accuracy. We need ~150-200k params to hit 85-90% accuracy while staying under 1 MB RAM.

## Model Variants

| Variant            | Parameters  | FP32 RAM    | INT4 Storage | Target Accuracy |
| ------------------ | ----------- | ----------- | ------------ | --------------- |
| Tiny               | 44,098      | ~172 KB     | ~55 KB       | 80-85%          |
| **Small**          | **175,234** | **~686 KB** | **~220 KB**  | **85-90%**      |
| Medium             | 421,570     | ~1.6 MB     | ~530 KB      | 88-92%          |
| Phase 2 (baseline) | 545,890     | ~2.1 MB     | ~276 KB      | 91.32%          |

All models use **CNN + global pooling** (no RNN/GRU) for simplicity and efficiency.

## Architecture Details

### Small Model (Recommended for Earbuds)

```
Input: [batch, 1, 80, time]

CNN Blocks:
  Conv2d(1→32) + BatchNorm + ReLU + MaxPool(2,2)    # 80 → 40
  Conv2d(32→64) + BatchNorm + ReLU + MaxPool(2,2)   # 40 → 20
  Conv2d(64→128) + BatchNorm + ReLU + MaxPool(2,2)  # 20 → 10

Global Pool:
  Mean over time dimension → [batch, 128×10 = 1280]

Classifier:
  Linear(1280 → 64) + ReLU + Dropout(0.3)
  Linear(64 → 2)
```

**Total: 175,234 parameters, ~686 KB FP32**

## Training Progress (Preliminary)

### Small Model — Direct Training on Hard Labels

Training interrupted after ~1 epoch (50 epochs planned):

| Epoch | Train Loss | Train Acc | Test Acc | Notes       |
| ----- | ---------- | --------- | -------- | ----------- |
| 1     | 0.6855     | 59.39%    | 59.11%   | First epoch |

Earlier run (same architecture, 40 epochs) reached:

- **Best: 88.72%** at epoch 12
- Oscillating between 68-89% in later epochs (overfitting signs)

### Small Model — Knowledge Distillation from Phase 2

Training interrupted after ~1 epoch:

| Epoch | Train Loss | Train Acc | Test Acc | Notes                     |
| ----- | ---------- | --------- | -------- | ------------------------- |
| 1     | 1.1623     | 65.59%    | 67.03%   | KD loss higher (expected) |

**Expected:** KD should provide smoother convergence and +1-3 pp improvement over direct
training, especially for multi-class extension.

## To Resume Training

### Small Model — Direct Training

```bash
# Train from scratch (50 epochs)
uv run src/scripts/train_phase4b.py \
  --model-size small \
  --mode direct \
  --epochs 50 \
  --batch-size 64 \
  --lr 0.001

# Or resume from checkpoint (if --resume flag is added)
uv run src/scripts/train_phase4b.py \
  --model-size small \
  --mode direct \
  --epochs 50 \
  --resume-from data/experiments/phase4b/small_direct/model_best.pth
```

### Small Model — Knowledge Distillation

```bash
uv run src/scripts/train_phase4b.py \
  --model-size small \
  --mode kd \
  --epochs 50 \
  --batch-size 64 \
  --lr 0.001 \
  --kd-alpha 0.5 \
  --temperature 2.0
```

### Medium Model — If Small Doesn't Hit 90%

```bash
uv run src/scripts/train_phase4b.py \
  --model-size medium \
  --mode direct \
  --epochs 50 \
  --batch-size 64
```

## Knowledge Distillation for Multi-Class Extension

When extending to 3+ languages (e.g., Danish, English, Swedish, Norwegian, German), KD
becomes more valuable:

**Why KD helps more with multiple classes:**

With 2 classes:

```
Teacher: [Danish: 0.92, English: 0.08]
→ Just tells student "how confident"
```

With 5 classes:

```
Teacher: [Danish: 0.60, Swedish: 0.25, Norwegian: 0.10, German: 0.03, English: 0.02]
→ Teaches class similarities (Scandinavian languages cluster together)
→ Student learns decision boundaries, not just labels
```

**Recommended for multi-class:**

- Train Phase 2 teacher on all 5 languages first
- Use KD with α=0.5-0.7, T=2.0-3.0
- Expect +2-5 pp improvement over direct training for compact students

## Deployment Guide

### Export to INT4 (Storage Optimization)

```python
import torch
from tiny_language_detection.models.tiny_cnn import create_small_cnn

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

# Export
model = create_small_cnn(num_languages=2)
model.load_state_dict(torch.load("data/experiments/phase4b/small_direct/model_best.pth"))
quantised, metadata = quantise_to_int4(model.state_dict())

torch.save({
    "model": quantised,
    "metadata": metadata,
    "config": {"n_mels": 80, "num_languages": 2},
}, "small_cnn_int4.pt")  # ~220 KB
```

### Load for Inference (CPU)

```python
import torch
from tiny_language_detection.models.tiny_cnn import create_small_cnn

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
checkpoint = torch.load("small_cnn_int4.pt")
state_dict = dequantise_int4(checkpoint["model"], checkpoint["metadata"])

model = create_small_cnn(num_languages=2)
model.load_state_dict(state_dict)  # Now ~686 KB in RAM
model.eval()

# Standard FP32 inference
output = model(input_tensor)
```

### Memory Budget Breakdown

| Component                 | Size           |
| ------------------------- | -------------- |
| Model weights (FP32)      | ~686 KB        |
| Activations (batch=1)     | ~50 KB         |
| Audio buffer (16-bit, 3s) | ~96 KB         |
| Mel spectrogram buffer    | ~25 KB         |
| **Total RAM**             | **~857 KB** ✅ |

**Fits within:**

- Earbuds (100 KB – 1 MB) ✅
- Headphones (500 KB – 1 MB) ✅

## Expected Results by Model Size

| Model   | Params | RAM (FP32) | Expected Accuracy (2-class) | Expected Accuracy (5-class) |
| ------- | ------ | ---------- | --------------------------- | --------------------------- |
| Tiny    | 44k    | ~172 KB    | 80-85%                      | 65-75%                      |
| Small   | 175k   | ~686 KB    | 85-90%                      | 75-85%                      |
| Medium  | 422k   | ~1.6 MB    | 88-92%                      | 82-88%                      |
| Phase 2 | 546k   | ~2.1 MB    | 91.32%                      | TBD                         |

**Recommendation:** Start with **Small + KD** for 3-5 language support. If accuracy <80%
on 5+ languages, move to Medium.

## Files

- `src/tiny_language_detection/models/tiny_cnn.py` — Model definitions
- `src/scripts/train_phase4b.py` — Training script (direct + KD modes)
- `data/experiments/phase4b/` — Experiment outputs

## Next Steps

1. Complete 50-epoch training for Small model (direct + KD)
2. Compare final accuracies — select best mode
3. If Small hits >88%, proceed to multi-class training
4. If Small <85%, train Medium model
5. Export best model to INT4 for deployment testing on target hardware
