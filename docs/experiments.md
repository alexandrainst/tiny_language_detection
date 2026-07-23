# Experiments: Danish vs English Language Detection

**Project:** Tiny Language Detection
**Repository:** <https://github.com/alexandrainst/tiny_language_detection>
**Dataset:** YODAS-Granary (23 languages, 415h train + 2.3k test balanced)

---

## Executive Summary

| Phase | Architecture         | Params | RAM     | Accuracy | Key Finding                            |
|-------|---------------------|--------|---------|----------|----------------------------------------|
| 1     | MFCC + CNN          | 56k    | 224 KB  | 81.7%    | Baseline: speaker-independent, 11.8 pp DA>EN gap |
| 2     | Log-Mel + CNN-RNN   | 546k   | 2.1 MB  | 90.86%   | +9.1 pp over Phase 1, but 9.7× larger  |
| 3     | Wavelet + CNN       | 56k    | 224 KB  | 60.67%   | ❌ Underperformed: wavelets not suitable |
| 4b    | Compact CNN (Small) | 175k   | 686 KB  | 96.76%   | ✅ **Recommended:** beats Phase 2 by 5.9 pp, 3× smaller |

**Phase 4b Small (KD)** is the recommended model for deployment:

- **96.76% accuracy** (highest achieved)
- **686 KB RAM** (fits earbud <1 MB target)
- **175k params** (simpler than CNN-RNN)
- **Balanced:** Danish 98.05%, English 95.10%

---

## Phase 1: MFCC + CNN Baseline

**Goal:** Establish speaker-independent baseline following edge-device literature.

### Architecture

- **Features:** 40 MFCC coefficients (25ms window, 10ms hop)
- **Model:** 3-layer CNN (56k params)
- **Input:** [batch, 1, 40, time]
- **Output:** Binary logits (Danish vs English)

### Data

- **Source:** Common Voice 26 (historical — now using YODAS-Granary)
- **Test set:** 1,729 clips (1h DA + 1h EN), 544 speakers, speaker-independent split
- **Duration groups:** 0–2s (N=374), 2–4s (N=684), 4–6s (N=468), 6+s (N=203)

### Results

| Metric                | Value   |
|----------------------|---------|
| Overall accuracy      | 81.7%   |
| Danish accuracy       | 86.9%   |
| English accuracy      | 75.1%   |
| Training accuracy     | 91.6%   |
| Test accuracy         | 81.7%   |
| Overfitting gap       | 9.9 pp  |

**Findings:**

1. **86% on short clips (0–2s)** — best performance
2. **11.8 pp DA>EN gap** — model biases toward Danish
3. **Mild overfitting** — 9.9 pp train/test gap

**Status:** ✅ Baseline established. Code: `src/scripts/train_cnn.py` (refactored for N-class)

---

## Phase 2: Log-Mel Spectrogram + CNN-RNN

**Goal:** Improve accuracy with richer features and temporal modelling.

### Architecture

- **Features:** Log Mel-spectrogram (80 bins, 25ms window, 10ms hop)
- **Model:** 3-layer CNN + unidirectional GRU (546k params)
- **Input:** [batch, 1, 80, time]
- **Output:** Binary logits

### Results

| Metric                | Value   | vs. Phase 1 |
|----------------------|---------|-------------|
| Overall accuracy      | 90.86%  | +9.1 pp ✅  |
| Danish accuracy       | 84.39%  | −2.5 pp     |
| English accuracy      | 99.21%  | +24.1 pp ✅ |
| Training accuracy     | 96.8%   | +5.2 pp     |
| Test accuracy         | 90.86%  | +9.1 pp     |

**Key findings:**

1. **English accuracy surged** (75% → 99%) — temporal modelling helped
2. **Danish dropped slightly** (87% → 84%) — shifted bias
3. **6+s clips improved** (80% → 91%) — RNN captured long-range structure
4. **9.7× parameter increase** — 546k vs 56k

**Status:** ✅ Best accuracy before Phase 4b. Code: `src/scripts/train_cnn_rnn.py`

---

## Phase 3: Wavelet Spectrogram + CNN

**Goal:** Explore Continuous Wavelet Transform (CWT) as alternative to MFCC/mel.

### Architecture

- **Features:** CWT with Ricker wavelet (48 scales, 10ms hop)
- **Model:** Same CNN as Phase 1 (56k params)
- **Input:** [batch, 1, 48, time]
- **Output:** Binary logits

### Implementation

- FFT-based CWT for efficiency
- Configurable hop_length (temporal downsampling)
- Mother wavelet: Ricker (Mexican hat)

### Results

| Metric                | Value   | vs. Phase 1 |
|----------------------|---------|-------------|
| Overall accuracy      | 60.67%  | −21 pp ❌   |
| Danish accuracy       | 43.63%  | −43 pp ❌   |
| English accuracy      | 82.65%  | +7.5 pp     |
| Training accuracy     | 69.36%  | −22 pp      |
| Test accuracy         | 60.67%  | −21 pp      |

**Key findings:**

1. **Catastrophic Danish failure** (87% → 44%) — model ignored Danish features
2. **Modest English gain** (75% → 83%) — not worth the Danish collapse
3. **Wavelet time resolution issue** — full 16kHz output caused OOM, fixed with hop_length downsampling

**Status:** ❌ **Not recommended.** Wavelets do not justify complexity for this task.
**Code removed:** Phase 3 scripts deleted (Common Voice-dependent).

---

## Phase 4b: Compact CNN

**Goal:** Achieve Phase 2+ accuracy with <1 MB RAM for edge deployment.

### Architecture

**CompactCNNLanguageDetector** — scalable CNN with global pooling (no RNN):

```python
# Small configuration (recommended)
channels = [32, 64, 128]   # 3 CNN blocks
hidden_size = 128          # Classifier hidden layer
Total: 175k params
```

**Training modes:**

1. **Direct:** Standard cross-entropy on hard labels
2. **Knowledge Distillation (KD):** Soft targets from Phase 2 teacher (α=0.7, T=2.5)

### Results

| Model                     | Params | RAM    | Acc    | Danish | English | Training |
|--------------------------|--------|--------|--------|--------|---------|----------|
| Phase 4b Small (Direct)  | 175k   | 686 KB | 95.78% | 92.09% | 97.75%  | Direct   |
| **Phase 4b Small (KD)**  | 175k   | 686 KB | **96.76%** | **98.05%** | **95.10%** | KD       |

**Comparison vs. Phase 2:**

| Metric            | Phase 2 | Phase 4b KD | Δ        |
|------------------|---------|-------------|----------|
| Accuracy          | 90.86%  | 96.76%      | +5.9 pp ✅ |
| Danish accuracy   | 84.39%  | 98.05%      | +13.7 pp ✅ |
| English accuracy  | 99.21%  | 95.10%      | −4.1 pp     |
| Parameters        | 546k    | 175k        | −68% ✅    |
| RAM               | 2.1 MB  | 686 KB      | −67% ✅    |

**Key findings:**

1. **Beat Phase 2 by 5.9 pp** — simpler architecture (no RNN) outperformed CNN-RNN
2. **KD added +0.98 pp** over direct training (95.78% → 96.76%)
3. **Reversed Danish bias** — DA 98% > EN 95% (vs. Phase 2: DA 84% < EN 99%)
4. **Fits earbud RAM budget** — 686 KB < 1 MB target

**Status:** ✅ **Recommended for deployment.**
**Code:** `src/scripts/train_cnn.py` with `--model-size small` and `--kd true`

---

## Phase 4b: Compression Experiments

**Goal:** Reduce storage/RAM further via quantisation.

### Method

- Post-training quantisation (no QAT)
- Evaluated BF16, FP16, INT8, INT4
- Baseline: Phase 4b KD model (96.76%)

### Results

| Precision | Storage | Compression | Accuracy | Δ vs. Baseline | Verdict |
|-----------|---------|-------------|----------|----------------|---------|
| FP32      | 686 KB  | 1.0×        | 96.76%   | —              | Baseline |
| **BF16**  | 343 KB  | 2.0×        | 96.65%   | −0.11 pp       | ✅ Near-lossless |
| INT8      | 175 KB  | 3.9×        | 95.66%   | −1.10 pp       | ✅ Practical |
| INT4      | 90 KB   | 7.6×        | 71.72%   | −25 pp ❌       | ❌ Catastrophic |

**Key findings:**

1. **BF16 is optimal** — 2× compression, 0.11 pp loss (within noise)
2. **INT8 viable** — 4× compression, 1.1 pp loss (acceptable for constrained devices)
3. **INT4 collapses** — 16-level quantisation too coarse for this architecture
4. **Storage vs. RAM distinction:** Quantisation reduces storage but not runtime RAM (dequantisation required on CPU)

**Recommendations:**

- **BF16:** Best for deployment (near-lossless, 2× smaller)
- **FP16:** Alternative for web (ONNX Runtime Web support, 39% RAM savings)
- **INT8:** Only if storage is critical (4× smaller, minor accuracy loss)
- **INT4:** Not viable without QAT

**Code:** `src/scripts/compress_model.py`

---

## Model Comparison Summary

| Model                   | Params | RAM     | Storage | Accuracy | Danish | English | Best For              |
|------------------------|--------|---------|---------|----------|--------|---------|-----------------------|
| Phase 1 CNN            | 56k    | 224 KB  | 224 KB  | 81.7%    | 86.9%  | 75.1%   | Historical baseline   |
| Phase 2 CNN-RNN        | 546k   | 2.1 MB  | 2.1 MB  | 90.86%   | 84.4%  | 99.2%   | High accuracy (large) |
| Phase 4b Small (KD)    | 175k   | 686 KB  | 686 KB  | **96.76%** | **98.1%** | **95.1%** | **Earbuds (<1 MB)**   |
| Phase 4b Small BF16    | 175k   | 686 KB  | 343 KB  | 96.65%   | 98.2%  | 94.7%   | Storage-constrained   |
| Phase 4b Small INT8    | 175k   | 686 KB  | 175 KB  | 95.66%   | 98.5%  | 92.1%   | Minimal storage       |

---

## Scripts

| Script                       | Purpose                        | Models          |
|-----------------------------|--------------------------------|-----------------|
| `train_cnn.py`              | Train Compact CNN (N languages) | Tiny/Small/Medium |
| `train_cnn_rnn.py`          | Train CNN-RNN (N languages)     | Configurable    |
| `compress_model.py`         | Quantise to BF16/INT8/INT4      | Phase 4b models |
| `demo_server.py`            | Flask server for web demo       | Phase 4b Small  |
| `build_and_upload_dataset.py` | Build YODAS-Granary dataset  | N/A             |

---

## Usage Examples

### Train Phase 4b Small (23 languages)

```bash
uv run src/scripts/train_cnn.py \
  --dataset saattrupdan/yodas-granary-language-detection \
  --use-hf \
  --model-size small \
  --channels 32 64 128 \
  --hidden-size 128 \
  --epochs 60 \
  --kd true \
  --kd-teacher-checkpoint data/experiments/phase2/model_best.pth
```

### Compress to BF16

```bash
uv run src/scripts/compress_model.py \
  --checkpoint data/experiments/phase4b/model_best.pth \
  --precision bf16
```

### Run Demo

```bash
uv run src/scripts/demo_server.py
# Open http://localhost:7860
```

---

## Dataset

**YODAS-Granary** (23 languages, Hugging Face Hub):

| Split | Samples | Duration | Format |
|-------|---------|----------|--------|
| Train  | 211k   | 415h     | Parquet shards (128 samples each) |
| Test   | 2,300  | 8.5h     | Balanced (100 per language) |

**Streaming:** No local storage required — audio streamed on demand.

**HF Repo:** `saattrupdan/yodas-granary-language-detection`
**Docs:** `docs/granary-dataset.md`

---

## Key Decisions

1. **Wavelets rejected** (Phase 3): 21 pp worse than MFCC baseline
2. **Compact CNN chosen** (Phase 4b): Beats CNN-RNN by 5.9 pp, 3× smaller
3. **Knowledge Distillation:** +0.98 pp over direct training
4. **BF16 quantisation:** Recommended (near-lossless, 2× storage reduction)
5. **INT4 rejected:** 25 pp accuracy collapse without QAT

---

## Next Steps

1. **Multi-class training** — 23-language model with class weighting
2. **Diverse data** — Collect non-CV audio (different mics, rooms, noise)
3. **Hardware testing** — Deploy to B&O target device, measure real-world latency/RAM
4. **ONNX export** — Web demo with FP16 model

---

**See also:**

- `docs/granary-dataset.md` — YODAS-Granary dataset documentation
- `docs/literature-survey.md` — Background research and citations
- `PLAN.md` — Project roadmap and milestones
