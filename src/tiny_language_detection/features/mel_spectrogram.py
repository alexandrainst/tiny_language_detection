"""Log Mel-spectrogram feature extraction.

Log Mel-spectrograms are widely used for audio classification tasks, providing a
time-frequency representation that captures both spectral content and temporal dynamics.
Unlike MFCCs which compress information into coefficients, Mel-spectrograms preserve the
full spectral envelope, making them suitable for CNN-RNN architectures that can learn
spatiotemporal patterns [Jiang et al., 2025; Cerna et al., 2023].

References:
    Jiang, X. et al. (2025). M3Net: Efficient Time-Frequency Integration Network With
    Mirror Attention For Audio Classification On Edge. AAAI.

    Cerna, P. et al. (2023). An IoT-Based Language Recognition System For Indigenous
    Languages Using Integrated CNN And RNN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
import torchaudio


@dataclass
class MelSpectrogramConfig:
    """Configuration for Log Mel-spectrogram extraction.

    Attributes:
        sample_rate: Audio sample rate in Hz.
        n_mels: Number of Mel filter banks (64-128 recommended).
        n_fft: FFT window size.
        hop_length: Hop length between frames.
        win_length: Window length (defaults to n_fft).
        f_min: Minimum frequency in Hz.
        f_max: Maximum frequency in Hz.

    Example:
        >>> config = MelSpectrogramConfig(n_mels=80)
        >>> config.log_mel_bins  # Property access
        80
    """

    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    win_length: int | None = None
    f_min: float = 0.0
    f_max: float | None = None

    @property
    def log_mel_bins(self) -> int:
        """Alias for n_mels for clarity."""
        return self.n_mels


class LogMelSpectrogramExtractor:
    """Extract Log Mel-spectrogram features from audio.

    This extractor computes Mel-scale spectrograms followed by a log compression,
    which better matches human auditory perception and improves model convergence.

    The output shape is `[n_mels, time_steps]` where `time_steps` depends on the
    audio duration and hop length.

    Example:
        >>> extractor = LogMelSpectrogramExtractor(n_mels=80)
        >>> waveform, sr = torchaudio.load("audio.wav")
        >>> features = extractor.extract(waveform, sr)
        >>> features.shape  # [80, time_steps]
    """

    def __init__(self, config: MelSpectrogramConfig | None = None) -> None:
        """Initialise the extractor.

        Args:
            config: Feature extraction configuration. Uses defaults if None.
        """
        self.config = config or MelSpectrogramConfig()

        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            win_length=self.config.win_length,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
        )

    def extract(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> npt.NDArray[np.floating]:
        """Extract Log Mel-spectrogram features from a waveform.

        Args:
            waveform: Audio waveform tensor of shape `[channels, samples]`.
            sample_rate: Sample rate of the input audio.

        Returns:
            Log Mel-spectrogram array of shape `[n_mels, time_steps]`.

        Raises:
            ValueError: If sample rate mismatch or invalid input shape.
        """
        if sample_rate != self.config.sample_rate:
            raise ValueError(
                f"Expected sample rate {self.config.sample_rate} Hz, "
                f"got {sample_rate} Hz"
            )

        if waveform.dim() != 2:
            raise ValueError(
                f"Expected 2D waveform [channels, samples], got {waveform.dim()}D"
            )

        # Compute Mel spectrogram
        mel_spec = self.transform(waveform)  # [channels, n_mels, time]

        # Take first channel (mono)
        mel_spec = mel_spec[0]  # [n_mels, time]

        # Log compression: log(x + eps) for numerical stability
        log_mel_spec = torch.log(mel_spec + 1e-8)

        return log_mel_spec.numpy()

    def get_feature_shape(self, duration_seconds: float) -> tuple[int, int]:
        """Compute the expected feature shape for a given duration.

        Args:
            duration_seconds: Duration of audio in seconds.

        Returns:
            Tuple of (n_mels, time_steps).
        """
        samples = int(duration_seconds * self.config.sample_rate)
        time_steps = (samples // self.config.hop_length) + 1
        return (self.config.n_mels, time_steps)


def extract_log_mel_spectrogram(
    waveform: torch.Tensor, sample_rate: int, config: MelSpectrogramConfig | None = None
) -> npt.NDArray[np.floating]:
    """Convenience function for extracting Log Mel-spectrogram features.

    Args:
        waveform: Audio waveform of shape `[channels, samples]`.
        sample_rate: Sample rate in Hz.
        config: Feature configuration. Uses defaults if None.

    Returns:
        Log Mel-spectrogram array of shape `[n_mels, time_steps]`.

    Example:
        >>> waveform = torch.randn(1, 16000)
        >>> features = extract_log_mel_spectrogram(waveform, sample_rate=16000)
        >>> features.shape
        (80, 101)
    """
    extractor = LogMelSpectrogramExtractor(config)
    return extractor.extract(waveform, sample_rate)
