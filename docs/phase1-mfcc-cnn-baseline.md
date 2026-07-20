# Phase 1: MFCC + CNN Baseline for Danish–English Language Detection

**Date:** 20 July 2026
**Author:** Dan Saattrup Smart
**Code:** <https://github.com/alexandrainst/tiny_language_detection>

## Abstract

We present a lightweight CNN baseline for binary language detection (Danish vs English)
using MFCC features. The model achieves **81.7% accuracy** on a balanced
speaker-independent test set (1h per language). Training accuracy reaches 91.6% after 30
epochs. Per-language accuracy: Danish 86.9%, English 75.1% — an 11.8 pp gap indicating
room for improvement in cross-language generalisation.

---

## 1. Introduction

Automatic language detection (LID) of audio is a prerequisite for multilingual speech
systems on edge devices. This work establishes a Phase 1 baseline following literature
recommendations for edge-compatible architectures: MFCC features with a shallow CNN
classifier. The approach prioritises low parameter count (~56k) and fast inference over
state-of-the-art accuracy, targeting eventual deployment on Raspberry Pi and headset
hardware.

---

## 2. Data

### 2.1 Source

Data sampled from **Common Voice 26** (June 2026 release):

| Language | Source file        | Total clips | Total hours | Speakers |
|----------|--------------------|-------------|-------------|----------|
| Danish   | cv-corpus-26.0-da  | 11,249      | 12.6        | 291      |
| English  | cv-corpus-26.0-en  | 1,901,978   | ~3,781,000  | ~50,000  |

English data was partially downloaded (~805k clips available); Danish fully extracted.

### 2.2 Sampling Strategy

**Speaker-independent split:**

1. Group all clips by speaker (`client_id`)
2. Assign 80% of speakers to train, 20% to test (no speaker overlap)
3. Stratify by language to maintain balance

**Target duration:** 1 hour per language in each split.

### 2.3 Test Set Statistics

| Language | Clips | Duration | Unique speakers |
|----------|-------|----------|-----------------|
| Danish   | 974   | 1.00h    | 50              |
| English  | 755   | 1.00h    | 494             |
| **Total**| 1,729 | 2.00h    | 544             |

Lower speaker count for Danish reflects smaller total pool (291 vs ~50k).

### 2.4 Preprocessing

All audio converted to:

- **Sample rate:** 16 kHz
- **Channels:** Mono (downmixed if stereo)
- **Format:** 16-bit PCM

Rationale: 16 kHz captures speech frequencies up to 8 kHz (Nyquist), standard for edge
audio [Liu, 2026], minimising compute and memory.

---

## 3. Method

### 3.1 Feature Extraction

**MFCC configuration:**

| Parameter     | Value   |
|---------------|---------|
| Coefficients  | 20      |
| Window length | 25 ms   |
| Hop length    | 10 ms   |
| Mel bands     | 40      |
| Sample rate   | 16 kHz  |

MFCCs computed via `torchaudio.transforms.MFCC`. No utterance-level normalisation.

**Output shape:** `[num_mfcc=20, time_steps]`, varies with duration (typical 50–400).

### 3.2 Model Architecture

**LanguageDetectionCNN** (56,194 trainable parameters):

```
Input: [batch, 20, time_steps, 1]

Block 1: Conv2d(1→16, k=3) + BatchNorm + ReLU + MaxPool(2)
Block 2: Conv2d(16→32, k=3) + BatchNorm + ReLU + MaxPool(2)
Block 3: Conv2d(32→64, k=3) + BatchNorm + ReLU + MaxPool(2)
Global Average Pooling → [batch, 64]
Dropout(0.5)
Linear(64, 2) → logits
```

Design choices:

- 3 convolutional blocks follow edge audio guidelines [Jiang et al., 2025]
- Global average pooling reduces parameters vs. fully connected layers
- Dropout(0.5) regularises final layer

**Literature basis:**

- MFCCs for edge: [Darvishi, 2026; Patil et al., 2024]
- Lightweight CNNs: [Cerna et al., 2023; Zhu et al., 2025]

### 3.3 Training

**Hyperparameters:**

| Parameter      | Value            |
|----------------|------------------|
| Epochs         | 30               |
| Batch size     | 64               |
| Learning rate  | 0.0005 (Adam)    |
| Loss           | CrossEntropyLoss |
| Class weights  | Yes (imbalanced) |
| Device         | CPU (M4 Max)     |

**Class weighting:** Addresses training imbalance (749 EN vs 941 DA samples):

```
w_c = N_total / (N_classes × N_c)
```

Result: DA weight ≈ 0.84, EN weight ≈ 1.18.

**Variable-length handling:** Utterances padded to max in batch via custom `collate_fn`.

### 3.4 Evaluation Protocol

- **Checkpoint:** Final epoch (no validation-based early stopping in Phase 1)
- **Metrics:** Overall accuracy, per-duration accuracy, per-language accuracy, confusion
  matrix
- **No class weighting** during evaluation — test set already balanced

---

## 4. Results

### 4.1 Training Progress

| Epoch | Train Loss | Train Accuracy |
|-------|------------|----------------|
| 1     | 0.64       | 67.2%          |
| 10    | 0.39       | 83.0%          |
| 20    | 0.31       | 87.9%          |
| 30    | 0.27       | 91.6%          |

Training accuracy plateaus ~epoch 25; loss continues decreasing, mild overfitting.

### 4.2 Test Set Performance

| Metric                       | Value  |
|------------------------------|--------|
| **Overall accuracy**         | 81.72% |
| Accuracy: 0–2s clips (N=374) | 86.08% |
| Accuracy: 2–4s clips (N=684) | 81.58% |
| Accuracy: 4–6s clips (N=468) | 81.41% |
| Accuracy: 6+s clips (N=203)  | 80.28% |
| **Danish accuracy (N=974)**  | 86.86% |
| **English accuracy (N=755)** | 75.10% |

**Confusion matrix** (rows = true, cols = predicted):

|           | Pred: DA | Pred: EN |
|-----------|----------|----------|
| True: DA  | 846      | 128      |
| True: EN  | 188      | 567      |

**Observations:**

1. **Duration effect:** Short clips (0–2s) perform best (86%), longer clips stable
   (~80–81%)
2. **Language asymmetry:** Danish classified 11.8 pp better than English
3. **Model bias:** 188 EN→DA errors vs 128 DA→EN — biases toward Danish predictions

### 4.3 Comparison to Literature

| Study                 | Task      | Accuracy | Notes                       |
|-----------------------|-----------|----------|-----------------------------|
| This work (Phase 1)   | DA vs EN  | 81.7%    | 56k params, edge-targeted   |
| Jiang et al. (2025)   | M3Net     | 97–98%   | Mirror attention, more params |
| Cerna et al. (2023)   | IoT LID   | ~90%     | CNN+RNN, indigenous langs   |
| Zhu et al. (2025)     | CNN LID   | ~85%     | MFCC features               |

Phase 1 within 5–15 pp of comparable lightweight baselines, with significantly fewer
parameters.

---

## 5. Discussion

### 5.1 Limitations

1. **English data partial:** Only ~805k of 1.9M clips available
2. **Single run:** No multiple seeds or cross-validation
3. **No validation set:** No early stopping or hyperparameter search
4. **Danish speaker pool:** Only 291 speakers limits test diversity (50 test speakers)

### 5.2 Future Work (Phase 2+)

**Immediate improvements:**

- CNN-RNN hybrid (GRU/LSTM after CNN) [Cerna et al., 2023]
- Data augmentation (SpecAugment, pitch/speed perturbation)
- Validation-based early stopping
- Speaker adversarial training

**Compression for edge:**

- Pruning [Mou & Milanova, 2024]
- Quantisation (INT8) [Bittner et al., 2025]
- Knowledge distillation

---

## 6. Reproducibility

### 6.1 Checkpoint and Code

- **Checkpoint:** `data/experiments/phase1/model.pth`
- **Config:** `data/experiments/phase1/config.json`
- **Metrics:** `data/experiments/phase1/evaluation.json`
- **Code:** <https://github.com/alexandrainst/tiny_language_detection>

### 6.2 Running the Experiment

```bash
# Install
git clone https://github.com/alexandrainst/tiny_language_detection
cd tiny_language_detection
make install

# Sample data (requires CV26 da/en in data/)
uv run src/scripts/sample_data.py --target-hours 1.0 --seed 42

# Train
uv run src/scripts/train.py --phase 1 --epochs 30 --batch-size 64 --lr 0.0005

# Evaluate
uv run src/scripts/evaluate.py --phase 1
```

### 6.3 Compute & Runtime Requirements

**Training / evaluation (development):**

| Stage    | Hardware   | Time    |
| -------- | ---------- | ------- |
| Sampling | M4 Max CPU | ~2 min  |
| Training | M4 Max CPU | ~25 min |
| Evaluate | M4 Max CPU | ~5 s    |

Raspberry Pi 5 training estimated ~3–4 hours for 30 epochs.

**Inference footprint (the deployment-relevant cost):**

| Metric                                | Value  |
| ------------------------------------- | ------ |
| Trainable parameters                  | 56,194 |
| Model size (fp32, `model.pth`)        | 227 KB |
| Weight memory (fp32)                  | 220 KB |
| Weight memory (INT8, projected)       | ~55 KB |
| CPU forward pass (1 thread, ~3s clip) | 6.0 ms |

Latency measured single-threaded on an M4 Max CPU (weights do not affect it). The
~220 KB fp32 footprint and single-digit-millisecond latency make this model comfortable
for microcontroller-class and Raspberry Pi targets, with ample headroom for real-time
streaming inference.

---

## 7. Conclusion

Phase 1 establishes a reproducible MFCC+CNN baseline achieving 81.7% accuracy on
speaker-independent Danish–English language detection. The model is small (56k
parameters), trains quickly on CPU, and provides foundation for Phase 2 improvements
(temporal modelling, augmentation) and Phase 3–4 compression for edge deployment.

The 11.8 pp gap between Danish and English accuracy warrants investigation — future work
should examine whether this stems from acoustic similarity, speaker diversity, or
training dynamics.

---

## References

- Bittner, M. et al. (2025). Pruning State Space Models For Efficient Raw Audio
  Classification. *EUSIPCO*.
- Cerna, P. et al. (2023). IoT-Based Language Recognition Using CNN And RNN.
- Darvishi, M. (2026). Embedded ML For Microcontroller-Class Edge Devices.
- Jiang, X. et al. (2025). M3Net: Efficient Audio Classification On Edge. *AAAI*.
- Liu, D. (2026). Survey On Lightweight Audio Classification For Edge Devices.
- Mou, A. & Milanova, M. (2024). Deep Learning Model-Compression For Edge Audio.
- Patil, M. et al. (2024). Edge Impulse: TinyML Language Classification.
- Zhu, Y.-C. et al. (2025). Speech Abusive Language Detection Using MFCC + CNN.
