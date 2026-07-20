# Phase 2: Log-Mel + CNN-RNN for Danish–English Language Detection

**Date:** 20 July 2026
**Author:** Dan Saattrup Smart
**Code:** <https://github.com/alexandrainst/tiny_language_detection>

## Abstract

We extend the Phase 1 baseline with richer features and temporal modelling: Log
Mel-spectrograms (80 bins) feeding a shallow CNN followed by a unidirectional GRU. The
model achieves **90.9% accuracy** on the same balanced, speaker-independent test set
(1h per language, 1,729 clips), a **+9.1 pp** improvement over the Phase 1 MFCC+CNN
baseline (81.7%). Per-language accuracy: Danish 84.4%, English 99.2%. The temporal
model markedly improves longer utterances (6+s: 80.3% -> 91.2%) but shifts the error
bias from Danish (Phase 1) to English (Phase 2). The cost is size: 545,890 parameters,
~9.7x the Phase 1 model, which motivates the compression work in Phases 3 and 6.

---

## 1. Introduction

Phase 1 established a 56k-parameter MFCC+CNN baseline at 81.7% accuracy, with a notable
11.8 pp gap favouring Danish and stable-but-modest performance across clip durations.
The literature suggests two complementary levers for improvement on edge-compatible
architectures: (1) Log Mel-spectrograms, which retain more spectral detail than
truncated MFCC cepstra, and (2) a recurrent layer after the CNN to model temporal
structure, expected to help longer utterances [Cerna et al., 2023; Ezilarasan et al.,
2026]. Phase 2 combines both while keeping the same data, split, and evaluation
protocol as Phase 1 for a controlled comparison.

---

## 2. Data

Data, sampling strategy, and the speaker-independent test set are **identical to Phase
1** (Common Voice 26, 16 kHz mono, seed 42). This isolates the effect of the feature
and architecture changes.

### 2.1 Test Set Statistics

| Language  | Clips | Duration | Unique speakers |
| --------- | ----- | -------- | --------------- |
| Danish    | 974   | 1.00h    | 50              |
| English   | 755   | 1.00h    | 494             |
| **Total** | 1,729 | 2.00h    | 544             |

Duration groups (shared with Phase 1): 0–2s (N=374), 2–4s (N=684), 4–6s (N=468), 6+s
(N=203).

---

## 3. Method

### 3.1 Feature Extraction

**Log Mel-spectrogram configuration:**

| Parameter     | Value                  |
| ------------- | ---------------------- |
| Mel bins      | 80                     |
| FFT window    | 400 (25 ms)            |
| Hop length    | 160 (10 ms)            |
| f_min / f_max | 0 Hz / Nyquist (8 kHz) |
| Sample rate   | 16 kHz                 |

Computed via `torchaudio.transforms.MelSpectrogram` followed by a log transform
(`log(mel + eps)`). Output shape `[n_mels=80, time_steps]`, time varying with duration.
Compared to Phase 1's 20 MFCC coefficients over 40 Mel bands, Phase 2 doubles the Mel
resolution (80 bins) and skips the DCT/cepstral truncation, preserving more spectral
detail for the CNN.

### 3.2 Model Architecture

**CNNRNNLanguageDetector** (545,890 trainable parameters):

```text
Input: [batch, 1, n_mels=80, time]

CNN feature extractor (3 blocks):
  Block 1: Conv2d(1->32,k=3) + BN + ReLU
           Conv2d(32->32,k=3) + BN + ReLU + MaxPool(2)
  Block 2: Conv2d(32->64,k=3) + BN + ReLU
           Conv2d(64->64,k=3) + BN + ReLU + MaxPool(2)
  Block 3: Conv2d(64->128,k=3) + BN + ReLU
           Conv2d(128->128,k=3) + BN + ReLU + MaxPool(2)
  -> spatial dims reduced 8x: [batch, 128, 10, time/8]

Reshape to sequence: [batch, time/8, 128*10 = 1280]

GRU (1 layer, hidden=64, unidirectional)
  -> last hidden state [batch, 64]

Classifier: Dropout(0.3) + Linear(64, 2) -> logits
```

Design choices:

- Two conv layers per block (vs one in Phase 1) with channels 32/64/128 give the CNN
  more capacity to extract frequency-local patterns before pooling.
- The GRU consumes the CNN feature map as a time sequence, modelling how spectral
  content evolves across the utterance — the key addition over Phase 1.
- Unidirectional (not bidirectional) GRU for edge inference efficiency.
- The final classifier uses only the last hidden state, so inference cost is dominated
  by the CNN and a single recurrent pass.

**Literature basis:** CNN+RNN hybrids for LID [Cerna et al., 2023]; temporal modelling
for edge audio [Ezilarasan et al., 2026].

### 3.3 Training

**Hyperparameters:**

| Parameter     | Phase 2          | Phase 1 (ref)    |
| ------------- | ---------------- | ---------------- |
| Epochs        | 30               | 30               |
| Batch size    | 32               | 64               |
| Learning rate | 0.001 (Adam)     | 0.0005 (Adam)    |
| Loss          | CrossEntropyLoss | CrossEntropyLoss |
| Class weights | Yes (imbalanced) | Yes              |
| Device        | MPS (M4 Max)     | CPU (M4 Max)     |

Log Mel-spectrograms are extracted on the fly per batch; variable-length utterances are
padded to the batch maximum via a custom `collate_fn`. Class weighting uses the same
`w_c = N_total / (N_classes * N_c)` scheme as Phase 1.

### 3.4 Evaluation Protocol

Identical to Phase 1: final-epoch checkpoint (no validation-based early stopping),
metrics = overall accuracy, per-duration accuracy, per-language accuracy, and confusion
matrix, evaluated on the balanced test set without class weighting.

---

## 4. Results

### 4.1 Training Progress

| Epoch | Train Loss | Train Accuracy |
| ----- | ---------- | -------------- |
| 1     | 0.70       | 56.0%          |
| 5     | 0.40       | 82.8%          |
| 10    | 0.18       | 93.4%          |
| 15    | 0.11       | 96.2%          |
| 20    | 0.10       | 96.8%          |
| 22    | 0.04       | 98.6% (peak)   |
| 30    | 0.05       | 98.4%          |

Training accuracy rises faster than Phase 1 and plateaus near 98% from ~epoch 22, with
some epoch-to-epoch variance (e.g. dips at epochs 14 and 29). Final train accuracy 98.4%
vs test 90.9% — a 7.5 pp generalisation gap, slightly tighter than Phase 1's 9.9 pp
despite the larger model.

### 4.2 Test Set Performance

| Metric                       | Phase 2 | Phase 1 |
| ---------------------------- | ------- | ------- |
| **Overall accuracy**         | 90.86%  | 81.72%  |
| Accuracy: 0–2s clips (N=374) | 83.54%  | 86.08%  |
| Accuracy: 2–4s clips (N=684) | 91.81%  | 81.58%  |
| Accuracy: 4–6s clips (N=468) | 91.52%  | 81.41%  |
| Accuracy: 6+s clips (N=203)  | 91.20%  | 80.28%  |
| **Danish accuracy (N=974)**  | 84.39%  | 86.86%  |
| **English accuracy (N=755)** | 99.21%  | 75.10%  |

**Confusion matrix** (rows = true, cols = predicted):

|          | Pred: DA | Pred: EN |
| -------- | -------- | -------- |
| True: DA | 822      | 152      |
| True: EN | 6        | 749      |

**Observations:**

1. **Overall gain:** +9.1 pp over Phase 1, driven overwhelmingly by English recall
   (75.1% -> 99.2%).
2. **Duration effect reversed:** Phase 1 was best on short clips and weakest on long
   ones; Phase 2 is the opposite — 6+s clips jump from 80.3% to 91.2%, while 0–2s clips
   dip from 86.1% to 83.5%. This is the expected signature of temporal modelling: the
   GRU has more sequence to work with on longer utterances.
3. **Bias flipped:** Phase 1 over-predicted Danish (188 EN->DA errors); Phase 2
   over-predicts English (152 DA->EN errors vs only 6 EN->DA). The model now almost
   never mistakes English for Danish but misclassifies ~16% of Danish clips as English.
4. **Gap direction changed:** the per-language gap widened slightly (11.8 -> 14.8 pp)
   but flipped sign, alongside the large overall improvement.

### 4.3 Comparison to Literature

| Study               | Task     | Accuracy | Notes                          |
| ------------------- | -------- | -------- | ------------------------------ |
| This work (Phase 2) | DA vs EN | 90.9%    | 546k params, Log-Mel + CNN-RNN |
| This work (Phase 1) | DA vs EN | 81.7%    | 56k params, MFCC + CNN         |
| Cerna et al. (2023) | IoT LID  | ~90%     | CNN+RNN, indigenous langs      |
| Jiang et al. (2025) | M3Net    | 97–98%   | Mirror attention, more params  |

Phase 2 now matches the ~90% CNN+RNN result of Cerna et al. (2023), at the cost of a
much larger parameter count than the Phase 1 baseline.

---

## 5. Discussion

### 5.1 What Improved and Why

The combination of higher-resolution Log-Mel features and a recurrent temporal model
lifted overall accuracy by 9.1 pp. The duration breakdown strongly supports the
temporal-modelling hypothesis: the largest gains are on the longest clips, where the
GRU has the most sequence to integrate. English recall in particular went from the
Phase 1 weak point (75%) to near-perfect (99%).

### 5.2 Limitations

1. **Model size:** 545,890 parameters (~9.7x Phase 1) is heavy for the target edge
   hardware. Most of the cost is the wide 1,280-dim CNN-to-GRU interface. This must be
   reduced before deployment (Phases 3 and 6).
2. **English bias:** the model over-predicts English (16% of Danish clips misclassified
   vs <1% of English). The test set is balanced, so this is a genuine decision-boundary
   skew, not a prevalence artefact.
3. **Short-clip regression:** 0–2s accuracy dropped 2.5 pp — the recurrent model adds
   little when there is little sequence, and may be slightly hurt by the wider feature
   interface.
4. **Single run, no validation set:** as in Phase 1, no multiple seeds, early stopping,
   or hyperparameter search.

### 5.3 Future Work

- **Rebalance the decision boundary:** revisit class weighting / threshold, or add a
  small validation set for calibration, to close the Danish–English asymmetry.
- **Shrink the CNN->GRU interface:** reduce channels or add pooling over the frequency
  axis before the GRU; try depthwise-separable convolutions (Phase 3).
- **Augmentation:** SpecAugment and speed/pitch perturbation to improve Danish recall.
- **Compression:** pruning, INT8 quantisation, and distillation for edge deployment
  (Phase 6).

---

## 6. Reproducibility

### 6.1 Checkpoint and Code

- **Checkpoint:** `data/experiments/phase2/model.pth`
- **Config:** `data/experiments/phase2/config.json`
- **Metrics:** `data/experiments/phase2/evaluation.json`
- **Code:** <https://github.com/alexandrainst/tiny_language_detection>

### 6.2 Running the Experiment

```bash
# Install
git clone https://github.com/alexandrainst/tiny_language_detection
cd tiny_language_detection
make install

# Sample data (requires CV26 da/en in data/); shared with Phase 1
uv run src/scripts/sample_data.py --target-hours 1.0 --seed 42

# Train
uv run src/scripts/train_phase2.py --epochs 30 --batch-size 32 --lr 0.001

# Evaluate
uv run src/scripts/evaluate_phase2.py
```

### 6.3 Compute Requirements

| Stage    | Hardware   | Time    |
| -------- | ---------- | ------- |
| Training | M4 Max MPS | ~30 min |
| Evaluate | M4 Max MPS | ~24 s   |

---

## 7. Conclusion

Phase 2 improves Danish–English language detection from 81.7% to 90.9% by pairing Log
Mel-spectrograms with a CNN-RNN temporal model, confirming the hypothesis that recurrent
modelling helps longer utterances. The gain comes with two caveats: the model is nearly
10x larger than the Phase 1 baseline, and its errors are now concentrated on Danish (an
English-prediction bias). Phase 3 (depthwise-separable CNN) and Phase 6 (compression)
should target the size problem, while rebalancing and augmentation should address the
Danish recall gap.

---

## References

- Cerna, P. et al. (2023). IoT-Based Language Recognition Using CNN And RNN.
- Ezilarasan, M. et al. (2026). Temporal Neural Models For Edge Audio Classification.
- Jiang, X. et al. (2025). M3Net: Efficient Audio Classification On Edge. *AAAI*.
