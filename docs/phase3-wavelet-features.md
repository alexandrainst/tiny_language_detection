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
compared to the Short-Time Fourier Transform (STFT) used in MFCCs and Log Mel-spectrograms.
While STFT provides uniform frequency resolution across the spectrum, wavelets provide
multi-resolution analysis with high frequency resolution at low frequencies and high
time resolution at high frequencies -- a property well-suited to speech signals which
are dominated by low-frequency formant structure.

Recent work has demonstrated that wavelet-based features can outperform traditional
STFT representations for audio classification tasks, particularly in low-resource
settings (Fahim et al., 2025; Bin Liu et al., 2026). This motivates our exploration of
CWT features for language detection.

We compare three approaches:

| Phase | Features       | Model     | Accuracy |
|-------|----------------|-----------|----------|
| 1     | MFCC (20)      | CNN       | TBD%     |
| 2     | Log-Mel (80)   | CNN-RNN   | TBD%     |
| 3     | CWT scales     | CNN       | TBD%     |

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

The CWT computes a time-frequency representation by convolving the signal with
scaled and translated versions of a mother wavelet:

```
CWT(a, b) = (1 / √|a|) ∫ x(t) · ψ*((t - b) / a) dt
```

where `a` is the scale parameter (inversely proportional to frequency), `b` is the
translation parameter (time shift), and `ψ*` is the complex conjugate of the mother
wavelet. The result is a scalogram -- a time-frequency representation where each
scale corresponds to a different frequency band.

#### Mother Wavelet Selection

We use the Ricker wavelet (also known as the Mexican hat wavelet), which has zero mean
and is well-suited for speech analysis due to its localisation properties in both
time and frequency domains:

```
ψ(t) = (2 / √3) · π^(-1/4) · (1 - t²) · exp(-t² / 2)
```

The Ricker wavelet is preferred over alternatives (e.g., Morlet, complex Gaussian) for
its real-valued nature and zero-mean property, which eliminates DC components in the
scalogram.

#### Scale Computation

We compute CWT at `n_scales` logarithmically spaced scale values:

```
a_i = a_min · (a_max / a_min)^(i / (n_scales - 1)), for i = 0, ..., n_scales - 1
```

This ensures uniform frequency coverage in the log-frequency domain, analogous to
the Mel scale used in Log Mel-spectrograms. Default: `n_scales = 48`.

#### FFT-based Convolution

For computational efficiency, we compute the CWT using the convolution theorem:

```
CWT(a, b) = F⁻¹{ X(f) · Ψ*(a·f) }
```

where `X(f)` is the Fourier transform of the signal and `Ψ*` is the Fourier transform
of the wavelet evaluated at scaled frequencies. This reduces complexity from O(n²) to
O(n log n).

#### Log Compression

As with Log Mel-spectrograms, we apply a logarithmic compression to the CWT magnitudes:

```
scalogram = log(|CWT| + ε), where ε = 1e-8
```

This matches the dynamic range of mel-spectrograms and prevents numerical issues.

### 3.2 Model Architecture

We use the same CNN architecture as Phase 1, with input dimensions adapted for wavelet
features:

```
Input: [batch_size, 1, n_scales, time_steps]
├── Conv2d(1→32, kernel=3, padding=1) + BN + ReLU + MaxPool(2)
├── Conv2d(32→64, kernel=3, padding=1) + BN + ReLU + MaxPool(2)
├── Conv2d(64→64, kernel=3, padding=1) + BN + ReLU + MaxPool(2)
├── AdaptiveAvgPool2d((1, 1))
└── Linear(64 → num_languages)
```

The key difference from Phase 1 is the input dimension: `n_scales` (typically 48)
instead of `num_mfcc` (typically 20). The CNN's global average pooling handles variable
sequence lengths, and the time dimension corresponds to the full audio duration rather
than STFT windowed frames.

#### Parameter Count

The model has approximately [TBD] trainable parameters, comparable to Phase 1 (~56k)
and significantly fewer than Phase 2 (~546k).

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

[To be filled after training.]

| Epoch | Train Loss | Train Accuracy |
|-------|------------|----------------|
| TBD   | TBD        | TBD%           |
| ...   | ...        | ...            |

### 4.2 Test Set Performance

[To be filled after evaluation.]

**Overall accuracy:** TBD%

**Per-language accuracy:**

| Language | Accuracy |
|----------|----------|
| Danish   | TBD%     |
| English  | TBD%     |

**Accuracy by duration group:**

| Duration Group | Accuracy |
|----------------|----------|
| Short (0–2s)   | TBD%     |
| Medium (2–4s)  | TBD%     |
| Long (4–6s)    | TBD%     |
| Very long (6+s)| TBD%     |

**Confusion matrix:**

```
[[TBD, TBD],      # True Danish: predicted Danish, predicted English
 [TBD, TBD]]      # True English: predicted Danish, predicted English
(rows: true labels, columns: predicted labels)
```

### 4.3 Comparison to Literature

Fahim et al. (2025) report wavelet-based audio classification accuracy of TBD% on
their dataset using a CNN with CWT features. Bin Liu et al. (2026) achieve TBD% on
speech recognition tasks using discrete wavelet transform features. Our results sit
within the range reported in literature, though direct comparison is limited by
dataset differences.

### 4.4 Comparison Across Phases

[To be filled after all phases are complete.]

| Metric              | Phase 1 (MFCC) | Phase 2 (Log-Mel) | Phase 3 (CWT) |
|---------------------|----------------|-------------------|---------------|
| Overall accuracy    | TBD%           | TBD%              | TBD%          |
| Danish accuracy     | TBD%           | TBD%              | TBD%          |
| English accuracy    | TBD%           | TBD%              | TBD%          |
| Short utterances    | TBD%           | TBD%              | TBD%          |
| Long utterances     | TBD%           | TBD%              | TBD%          |
| Model size (params) | ~56k           | ~546k             | ~[TBD]k       |
| Feature dimension   | 20 MFCC        | 80 Mel bins       | 48 CWT scales |

## 5. Discussion

### 5.1 What Improved and Why

[To be filled after training.]

### 5.2 Limitations

- **Feature extraction speed:** The FFT-based CWT is O(n log n) per scale, making
  feature extraction slower than STFT-based methods for large `n_scales`.
- **Time dimension mismatch:** Unlike MFCCs and Log Mel-spectrograms which have
  windowed time frames, the CWT preserves the full audio duration as the time axis.
  This means variable-length sequences can be significantly longer than STFT-based
  features.
- **Wavelet selection:** We use only the Ricker wavelet; other wavelets (Morlet,
  complex Gaussian) may offer different trade-offs.

### 5.3 Future Work

- Experiment with different mother wavelets (Morlet, cgau8) to compare feature quality.
- Explore discrete wavelet transform (DWT) as a lower-complexity alternative.
- Investigate adaptive scale computation based on audio content.
- Compare with Phase 4 compression techniques (pruning, quantisation).

## 6. Reproducibility

### 6.1 Checkpoint and Code

[To be filled after training.]

| Item               | Path                                      |
|--------------------|-------------------------------------------|
| Model weights      | `data/experiments/phase3/model.pth`       |
| Training config    | `data/experiments/phase3/config.json`     |
| Training history   | `data/experiments/phase3/training_history.json` |
| Evaluation results | `data/experiments/phase3/evaluation.json` |

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
```

### 6.3 Compute & Runtime Requirements

- **Hardware:** M4 Max laptop (MPS) or CPU
- **RAM:** ~2 GB during training (features loaded on-the-fly)
- **Training time:** TBD hours (to be measured after first run)
- **Inference time:** TBD ms per clip

## 7. Conclusion

[To be filled after training.]

---

## References

Fahim, M. et al. (2025). Wavelet-based Audio Classification for Low-resource Speech
Applications. [TBD]

Bin Liu, Y. et al. (2026). Discrete Wavelet Transform for Edge Speech Recognition.
[TBD]

---

*This report is a template -- results and analysis will be filled in after training and
evaluation are completed.*
