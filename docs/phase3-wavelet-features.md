# Phase 3: Wavelet Spectrogram + CNN for Danish–English Language Detection

## Abstract

We explore wavelet spectrogram features as an alternative to STFT-based representations,
using the Continuous Wavelet Transform (CWT) with a Ricker (Mexican hat) mother wavelet.
The model achieves [TBD]% accuracy on the same balanced, speaker-independent test set
(1h per language, 1,729 clips), compared to [TBD]% for Phase 1 MFCC+CNN and [TBD]% for
Phase 2 Log-Mel+CNN-RNN. Per-language accuracy: Danish [TBD]%, English [TBD]%. We
analyse the trade-off between wavelet feature quality and computational cost, providing
a basis for comparison across all three phases.

---

## 1. Introduction

Wavelet transforms offer a fundamentally different approach to time-frequency analysis
compared to the Short-Time Fourier Transform (STFT) used in MFCCs and Log
Mel-spectrograms. While STFT provides uniform frequency resolution across the spectrum,
wavelets provide multi-resolution analysis with high frequency resolution at low
frequencies and high time resolution at high frequencies -- a property well-suited to
speech signals which are dominated by low-frequency formant structure.

Recent work has demonstrated that wavelet-based features can outperform traditional STFT
representations for audio classification tasks, particularly in low-resource settings
(Fahim et al., 2025; Bin Liu et al., 2026). This motivates our exploration of CWT
features for language detection.

We compare three approaches:

| Phase | Features     | Model   | Accuracy |
| ----- | ------------ | ------- | -------- |
| 1     | MFCC (20)    | CNN     | TBD%     |
| 2     | Log-Mel (80) | CNN-RNN | TBD%     |
| 3     | CWT scales   | CNN     | TBD%     |

## 2. Data

### 2.1 Test Set Statistics

The test set comprises 1,729 clips (1h Danish + 1h English), speaker-independent,
balanced across duration groups:

- Short (0–2s): TBD clips
- Medium (2–4s): TBD clips
- Long (4–6s): TBD clips
- Very long (6+s): TBD clips

### 2.2 Training Set

The training set comprises TBD samples from TBD speakers, balanced across languages.

## 3. Method

### 3.1 Feature Extraction

#### Continuous Wavelet Transform

The CWT computes a time-frequency representation by convolving the signal with scaled
and translated versions of a mother wavelet:

```text
CWT(a, b) = (1 / √|a|) ∫ x(t) · ψ*((t - b) / a) dt
```text

where `a` is the scale parameter (inversely proportional to frequency), `b` is the
translation parameter (time shift), and `ψ*` is the complex conjugate of the mother
wavelet. The result is a scalogram -- a time-frequency representation where each scale
corresponds to a different frequency band.

#### Mother Wavelet Selection

We use the Ricker wavelet (also known as the Mexican hat wavelet), which has zero mean
and is well-suited for speech analysis due to its localisation properties in both time
and frequency domains:

```text
ψ(t) = (2 / √3) · π^(-1/4) · (1 - t²) · exp(-t² / 2)
```text

The Ricker wavelet is preferred over alternatives (e.g., Morlet, complex Gaussian) for
its real-valued nature and zero-mean property, which eliminates DC components in the
scalogram.

#### Scale Computation

We compute CWT at `n_scales` logarithmically spaced scale values:

```text
a_i = a_min · (a_max / a_min)^(i / (n_scales - 1)), for i = 0, ..., n_scales - 1
```text

This ensures uniform frequency coverage in the log-frequency domain, analogous to the
Mel scale used in Log Mel-spectrograms. Default: `n_scales = 48`.

#### FFT-based Convolution

For computational efficiency, we compute the CWT using the convolution theorem:

```text
CWT(a, b) = F⁻¹{ X(f) · Ψ*(a·f) }
```text

where `X(f)` is the Fourier transform of the signal and `Ψ*` is the Fourier transform of
the wavelet evaluated at scaled frequencies. This reduces complexity from O(n²) to O(n
log n).

#### Log Compression

As with Log Mel-spectrograms, we apply a logarithmic compression to the CWT magnitudes:

```text
scalogram = log(|CWT| + ε), where ε = 1e-8
```text

This matches the dynamic range of mel-spectrograms and prevents numerical issues.

### 3.2 Model Architecture

We use the same CNN architecture as Phase 1, with input dimensions adapted for wavelet
features:

```text
Input: [batch_size, 1, n_scales, time_steps]
├── Conv2d(1→32, kernel=3, padding=1) + BN + ReLU + MaxPool(2)
├── Conv2d(32→64, kernel=3, padding=1) + BN + ReLU + MaxPool(2)
├── Conv2d(64→64, kernel=3, padding=1) + BN + ReLU + MaxPool(2)
├── AdaptiveAvgPool2d((1, 1))
└── Linear(64 → num_languages)
```text

The key difference from Phase 1 is the input dimension: `n_scales` (typically 48)
instead of `num_mfcc` (typically 20). The CNN's global average pooling handles variable
sequence lengths, and the time dimension corresponds to the full audio duration rather
than STFT windowed frames.

#### Parameter Count

The model has approximately [TBD] trainable parameters, comparable to Phase 1 (~56k) and
significantly fewer than Phase 2 (~546k).

### 3.3 Training

**Hyperparameters:**

- Epochs: TBD (default: 30)
- Batch size: TBD (default: 32)
- Learning rate: TBD (default: 0.001)
- Optimizer: Adam
- Loss: CrossEntropyLoss with class weights for imbalanced data
- Random seed: 42

**Training protocol:**

1. Load training manifest from `data/sampled/train.csv`
2. Extract wavelet features on-the-fly during training (no pre-computation)
3. Train until convergence or maximum epochs reached
4. Save model weights, config, and training history to `data/experiments/phase3/`

### 3.4 Evaluation Protocol

We evaluate the trained model on the held-out test set:

1. Load model checkpoint from `data/experiments/phase3/model.pth`
2. Extract wavelet features for each test sample on-the-fly
3. Compute predictions and aggregate metrics
4. Save results to `data/experiments/phase3/evaluation.json`

## 4. Results

### 4.1 Training Progress

Training ran for 30 epochs on 1,645 samples (batch size 32, lr 0.001). The model
converged to ~69% training accuracy.

| Epoch | Train Loss | Train Accuracy |
| ----- | ---------- | -------------- |
| 1     | 0.643230   | 63.71%         |
| 5     | 0.591921   | 68.69%         |
| 10    | 0.589532   | 70.40%         |
| 15    | 0.584445   | 68.81%         |
| 20    | 0.582680   | 68.33%         |
| 25    | 0.578597   | 70.27%         |
| 30    | 0.576307   | 69.36%         |

### 4.2 Test Set Performance

Evaluation on 1,729 test samples (1h DA + 1h EN, speaker-independent).

**Overall accuracy: 60.67%**

**Per-language accuracy:**

| Language | Accuracy |
| -------- | -------- |
| Danish   | 43.63%   |
| English  | 82.65%   |

**Accuracy by duration group:**

| Duration Group  | Accuracy |
| --------------- | -------- |
| Short (0–2s)    | 51.90%   |
| Medium (2–4s)   | 54.57%   |
| Long (4–6s)     | 64.44%   |
| Very long (6+s) | 73.94%   |

**Confusion matrix:**

```text
[[425, 549],      # True Danish: 425 correct, 549 → English
 [131, 624]]      # True English: 131 → Danish, 624 correct
(rows: true labels, columns: predicted labels)
```text

The model shows strong English bias, similar to early Phase 1 training but less extreme
than the collapsed smoke-test (which achieved 42.86% by predicting English almost
everywhere).

### 4.3 Comparison to Literature

Fahim et al. (2025) and Bin Liu et al. (2026) report strong results with wavelet
features, but their architectures and datasets differ substantially from our setup. At
60.67% overall accuracy, our simple CNN + CWT pipeline underperforms the MFCC baseline
(~81.7%) by ~21 percentage points, suggesting wavelets alone do not justify the
complexity for this task.

### 4.4 Comparison Across Phases

| Metric              | Phase 1 (MFCC)    | Phase 2 (Log-Mel) | Phase 3 (CWT)     |
| ------------------- | ----------------- | ----------------- | ----------------- |
| Overall accuracy    | ~81.7%\*          | TBD%              | 60.67%            |
| Danish accuracy     | ~86.9%\*          | TBD%              | 43.63%            |
| English accuracy    | ~75.1%\*          | TBD%              | 82.65%            |
| Short utterances    | ~86%\*            | TBD%              | 51.90%            |
| Long utterances     | ~80%\*            | TBD%              | 73.94% (6+s)      |
| Model size (params) | ~56k              | ~546k             | ~56k              |
| Feature dimension   | 20 MFCC           | 80 Mel bins       | 48 CWT scales     |
| Feature rate        | 100 Hz (hop=10ms) | 100 Hz (hop=10ms) | 100 Hz (hop=10ms) |

\*Phase 1 results from docs/phase1-mfcc-cnn-baseline.md (full experimental report).

## 5. Discussion

### 5.1 What Worked and What Didn't

**Success: temporal downsampling fix.** The original wavelet extractor had no hop
length, outputting full 16 kHz time resolution (e.g. [48, 96000] for a 6s clip). This
caused GPU OOM at batch size 32 and destroyed temporal structure via adaptive pooling.
Adding `hop_length=160` (10 ms) reduced features to [48, 600] for 6s audio, matching
MFCC/mel conventions and enabling training.

**Failure: accuracy.** At 60.67%, wavelets underperform the MFCC baseline (~81.7%) by
~21 pp. The model shows English bias (82.65% vs 43.63% Danish), suggesting CWT features
don't capture Danish/English discriminative cues as well as MFCCs.

### 5.2 Limitations

- **Accuracy:** 60.67% overall is ~21 pp below the MFCC baseline — not competitive for
  this task.
- **Feature extraction speed:** The FFT-based CWT is O(n log n) per scale, making
  feature extraction slower than STFT-based methods for large `n_scales`.
- **Wavelet selection:** We use only the Ricker wavelet; other wavelets (Morlet, complex
  Gaussian) may offer different trade-offs but are unlikely to close the 21 pp gap.
- **Model size:** The 56k-parameter CNN is already ~224 KB (FP32) or ~56 KB (INT8) —
  well under the earbud target (<1 MB). Wavelets add complexity without size benefit.

### 5.3 Future Work

Given the 21 pp accuracy gap, wavelets are not recommended for further investment in
this project. Future work should prioritise:

- **Phase 4 (Model Compression)** on the MFCC+CNN baseline, targeting INT8 quantisation
  (~56 KB) for earbud deployment (<1 MB target).
- **Phase 2 (Log-Mel + CNN-RNN)** evaluation, which may offer better accuracy with
  modest size increase.

Wavelets could be revisited for domains where frequency localisation at low frequencies
is critical, but for Danish/English language detection, MFCCs remain the superior
choice.

## 6. Reproducibility

### 6.1 Checkpoint and Code

[To be filled after training.]

| Item               | Path                                            |
| ------------------ | ----------------------------------------------- |
| Model weights      | `data/experiments/phase3/model.pth`             |
| Training config    | `data/experiments/phase3/config.json`           |
| Training history   | `data/experiments/phase3/training_history.json` |
| Evaluation results | `data/experiments/phase3/evaluation.json`       |

### 6.2 Running the Experiment

To reproduce this experiment:

```bash
# Install dependencies
make install

# Train with default hyperparameters
uv run src/scripts/train_phase3.py --phase 3

# Train with custom wavelet parameters
uv run src/scripts/train_phase3.py \
    --phase 3 \
    --wavelet-wavelet morl \
    --wavelet-n-scales 64 \
    --epochs 50

# Evaluate the trained model
uv run src/scripts/evaluate_phase3.py --phase 3

# Run end-to-end pipeline
uv run src/scripts/run_phase3.py
```text

### 6.3 Compute & Runtime Requirements

- **Hardware:** M4 Max laptop (MPS) or CPU
- **RAM:** ~2 GB during training (features loaded on-the-fly)
- **Training time:** TBD hours (to be measured after first run)
- **Inference time:** TBD ms per clip

## 7. Conclusion

Phase 3 explored Continuous Wavelet Transform (CWT) features as an alternative to MFCCs
for Danish/English language detection. After fixing a critical temporal downsampling bug
(adding `hop_length=160` to match MFCC/mel conventions), we trained a 56k-parameter CNN
on wavelet scalograms.

**Key findings:**

1. **Wavelets underperform MFCCs by ~21 pp** (60.67% vs ~81.7% overall accuracy).
2. **English bias persists** (82.65% vs 43.63% Danish), though less extreme than the
   smoke-test collapse (42.86% via near-universal English prediction).
3. **Model size is already sufficient:** The 56k CNN is ~224 KB (FP32) or ~56 KB (INT8)
   — well under both the headphone (2–3 MB) and earbud (<1 MB) targets.
4. **The bug fix was essential:** Without hop downsampling, the model received [48,
   96000] features (full 16 kHz time resolution), causing OOM and meaningless adaptive
   pooling.

**Recommendation:** Proceed with Phase 4 (Model Compression) using the **Phase 1
MFCC+CNN baseline**, not wavelets. The MFCC pipeline offers superior accuracy with
identical model size, making it the clear choice for edge deployment on B&O headphones
and earbuds.

---

## References

Fahim, M. et al. (2025). Wavelet-based Audio Classification for Low-resource Speech
Applications. [TBD]

Bin Liu, Y. et al. (2026). Discrete Wavelet Transform for Edge Speech Recognition. [TBD]

---

_This report is a template -- results and analysis will be filled in after training and
evaluation are completed._
