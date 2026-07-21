"""Log Mel-spectrogram and wavelet feature extraction.

Log Mel-spectrograms are widely used for audio classification tasks, providing a
time-frequency representation that captures both spectral content and temporal dynamics.
Unlike MFCCs which compress information into coefficients, Mel-spectrograms preserve the
full spectral envelope, making them suitable for CNN-RNN architectures that can learn
spatiotemporal patterns [Jiang et al., 2025; Cerna et al., 2023].

Wavelet transforms offer an alternative time-frequency representation with superior
frequency localisation at low frequencies and temporal localisation at high frequencies.
The Continuous Wavelet Transform (CWT) with a Ricker wavelet has shown promise for
language detection tasks [Fahim et al., 2025; Bin Liu et al., 2026].

References:
    Jiang, X. et al. (2025). M3Net: Efficient Time-Frequency Integration Network With
    Mirror Attention For Audio Classification On Edge. AAAI.

    Cerna, P. et al. (2023). An IoT-Based Language Recognition System For Indigenous
    Languages Using Integrated CNN And RNN.

    Fahim, M. et al. (2025). Wavelet-based Audio Classification for Low-resource
    Speech Applications.

    Bin Liu, Y. et al. (2026). Discrete Wavelet Transform for Edge Speech Recognition.
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
        >>> config = MelSpectrogramConfig(n_mels=80)
        >>> extractor = LogMelSpectrogramExtractor(config)
        >>> waveform = torch.randn(1, 16000)  # 1s at 16kHz
        >>> features = extractor.extract(waveform, sample_rate=16000)
        >>> features.shape[0]  # [80, time_steps]
        80
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


# ---------------------------------------------------------------------------
# Mother wavelet spectrum functions (Fourier transforms of mother wavelets)
# ---------------------------------------------------------------------------


def _ricker_spectrum(freqs: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Return the Fourier transform of the Ricker (Mexican hat) wavelet.

    The Ricker wavelet is defined as:
        ψ(t) = (1 - 2π²t²) · exp(-π²t²)

    Its Fourier transform is:
        Ψ(f) = πf² · exp(-π²f²/4)

    Args:
        freqs:
          Array of frequencies in Hz.

    Returns:
        Wavelet spectrum values at each frequency.
    """
    pi2 = np.pi * np.pi
    return pi2 * freqs**2 * np.exp(-pi2 * freqs**2 / 4)


def _morl_spectrum(freqs: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Return the Fourier transform of the Morse wavelet.

    The Morse wavelet is parameterised by β (time-bandwidth product) and γ.
    For audio applications, β=30 and γ=3 are typical values.

    Its Fourier transform (for real-valued case, positive frequencies only):
        Ψ(f) = πf^{γ-1} · exp(-πβ · f - i·sign(f)·π/2)

    For the real part at positive frequencies:
        Re[Ψ(f)] = πf^{γ-1} · exp(-πβ · f) · cos(π/2)
                = 0

    We use the magnitude spectrum for the CWT:
        |Ψ(f)| = πf^{γ-1} · exp(-πβ · f)

    Args:
        freqs:
          Array of frequencies in Hz.

    Returns:
        Wavelet spectrum values at each frequency.
    """
    beta = 30.0
    gamma = 3.0
    pi2 = np.pi * np.pi
    # Magnitude: π·f^{γ-1} · exp(-πβ·f)
    return np.pi * freqs ** (gamma - 1) * np.exp(-pi2 * beta * freqs)


def _cgau8_spectrum(freqs: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Return the Fourier transform of the complex Gaussian wavelet.

    The complex Gaussian wavelet is defined as:
        ψ(t) = (2π)^{-1/4} · exp(-t²/2 + i·ω₀·t)

    Its Fourier transform is:
        Ψ(f) = (2π)^{1/4} · exp(-(f - ω₀)²/2)

    For the real part at frequency f:
        Re[Ψ(f)] = (2π)^{1/4} · cos(ω₀·t) · exp(-f²/2)

    We use the magnitude spectrum for the CWT:
        |Ψ(f)| = (2π)^{1/4} · exp(-(f - ω₀)²/2)

    where ω₀ is the central frequency.

    Args:
        freqs:
          Array of frequencies in Hz.

    Returns:
        Wavelet spectrum values at each frequency.
    """
    omega0 = 5.0
    sqrt2pi4 = np.sqrt(2 * np.pi) ** 0.25
    return sqrt2pi4 * np.exp(-((freqs - omega0) ** 2) / 2)


def _mexh_spectrum(freqs: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Return the Fourier transform of the Mexican hat wavelet.

    The Mexican hat (Ricker) wavelet is identical to the Ricker wavelet:
        ψ(t) = (2/√3) · π^{-1/4} · (1 - t²) · exp(-t²/2)

    Its Fourier transform is:
        Ψ(f) = √(2/π) · f² · exp(-f²/2)

    Args:
        freqs:
          Array of frequencies in Hz.

    Returns:
        Wavelet spectrum values at each frequency.
    """
    return np.sqrt(2 / np.pi) * freqs**2 * np.exp(-(freqs**2) / 2)


class WaveletSpectrogramConfig:
    """Configuration for wavelet spectrogram extraction.

    Uses the Continuous Wavelet Transform (CWT) with a Ricker (Mexican hat)
    wavelet as the mother function. The CWT produces a time-frequency
    representation called a scalogram, which captures multi-scale frequency
    content with better localisation than short-time Fourier transforms
    at low frequencies.

    Attributes:
        sample_rate:
          Audio sample rate in Hz.
        widths:
          Array of scales to evaluate the wavelet transform at. Higher values
          correspond to lower frequencies. Typically 32-64 for audio.
        wavelet:
          Name of the mother wavelet. Default is "ricker" (Mexican hat),
          which has zero mean and is well-suited for speech analysis.
        f_min:
          Minimum frequency in Hz for scale computation. Defaults to 0 Hz.
        f_max:
          Maximum frequency in Hz for scale computation. Defaults to
          Nyquist frequency (sample_rate / 2).
        hop_length:
          Number of samples between consecutive pooled time frames. The
          magnitude scalogram is mean-pooled along the time axis into
          non-overlapping windows of this size. Defaults to 10 ms in
          samples (160 at 16 kHz).

    Example:
        >>> config = WaveletSpectrogramConfig(widths=48)
        >>> config.widths.shape  # Array of scales
        (48,)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        widths: int | npt.NDArray[np.integer] = 48,
        wavelet: str = "ricker",
        f_min: float = 0.0,
        f_max: float | None = None,
        hop_length: int | None = None,
    ) -> None:
        """Initialise the configuration.

        Args:
            sample_rate:
              Audio sample rate in Hz. Defaults to 16 kHz.
            widths:
              Number of scales or array of scale values. Defaults to 48 scales.
            wavelet:
              Mother wavelet name for CWT. Supported: "ricker", "morl" (Morse),
              "cgau8" (complex Gaussian). Defaults to "ricker".
            f_min:
              Minimum frequency in Hz for scale computation. Defaults to 0 Hz.
            f_max:
              Maximum frequency in Hz for scale computation. Defaults to
              Nyquist frequency.
            hop_length (optional):
              Number of samples between consecutive pooled time frames. The
              magnitude scalogram is mean-pooled along the time axis into
              non-overlapping windows of this size. Defaults to 10 ms in
              samples (160 at 16 kHz).

        Raises:
            ValueError:
              If wavelet name is unsupported, widths is invalid, or
              hop_length is less than 1.
        """
        self.sample_rate = sample_rate
        self.wavelet = wavelet
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate / 2

        if hop_length is None:
            hop_length = int(sample_rate * 10 / 1000)
        if hop_length < 1:
            raise ValueError(f"hop_length must be at least 1, got {hop_length}")
        self.hop_length = hop_length

        # Validate wavelet name
        valid_wavelets = {"ricker", "morl", "cgau8", "mexh"}
        if wavelet not in valid_wavelets:
            raise ValueError(
                f"Unsupported wavelet '{wavelet}'. "
                f"Valid options: {sorted(valid_wavelets)}"
            )

        # Convert widths to array of scale values
        self.widths = self._compute_widths(widths)

    def _compute_widths(
        self, widths: int | npt.NDArray[np.integer]
    ) -> npt.NDArray[np.floating]:
        """Compute or validate the array of scale values.

        Args:
            widths:
              Number of scales (integer) or pre-computed array of scale values.

        Returns:
            Array of scale values for CWT computation.

        Raises:
            ValueError: If widths is not a positive integer or valid array.
        """
        if isinstance(widths, int):
            if widths < 1:
                raise ValueError(f"widths must be a positive integer, got {widths}")
            # Use logarithmically spaced scales for uniform frequency coverage
            # in log-frequency domain, similar to Mel scale behaviour.
            return np.logspace(
                start=np.log10(self.f_min + 1), stop=np.log10(self.f_max), num=widths
            )
        if isinstance(widths, np.ndarray):
            if widths.ndim != 1:
                raise ValueError(f"widths must be a 1D array, got {widths.ndim}D")
            if len(widths) < 1:
                raise ValueError("widths array must not be empty")
            return widths.astype(np.float64)
        raise ValueError(f"widths must be int or ndarray, got {type(widths).__name__}")

    @property
    def n_scales(self) -> int:
        """Number of frequency scales in the wavelet transform."""
        return len(self.widths)


class WaveletSpectrogramExtractor:
    """Extract wavelet spectrogram features from audio using CWT.

    Computes the Continuous Wavelet Transform (CWT) of an audio waveform,
    producing a time-frequency representation called a scalogram. The Ricker
    wavelet is used as the mother function, which provides good localisation
    in both time and frequency domains for speech analysis.

    The output shape is `[n_scales, time_steps]` where `time_steps` is
    `ceil(n_samples / hop_length)` after temporal mean-pooling, depending on
    the audio duration, sampling rate, and hop length.

    References:
        Fahim, M. et al. (2025). Wavelet-based Audio Classification for
        Low-resource Speech Applications.

        Bin Liu, Y. et al. (2026). Discrete Wavelet Transform for Edge
        Speech Recognition.

    Example:
        >>> config = WaveletSpectrogramConfig(widths=48)
        >>> extractor = WaveletSpectrogramExtractor(config)
        >>> waveform = torch.randn(1, 16000)  # 1 second at 16 kHz
        >>> features = extractor.extract(waveform, sample_rate=16000)
        >>> features.shape[0]  # [48 scales, time_steps]
        48
    """

    def __init__(self, config: WaveletSpectrogramConfig | None = None) -> None:
        """Initialise the extractor.

        Args:
            config:
              Feature extraction configuration. Uses defaults if None.
        """
        self.config = config or WaveletSpectrogramConfig()

    def extract(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> npt.NDArray[np.floating]:
        """Extract wavelet spectrogram features from a waveform.

        Computes the Continuous Wavelet Transform (CWT) using FFT-based
        convolution. For each scale in widths, the CWT computes:

            CWT(a, b) = F^{-1}{ X(f) · Ψ*(a·f) }

        where X(f) is the Fourier transform of the signal and Ψ* is the
        complex conjugate of the wavelet's Fourier transform evaluated at
        frequency a·f.

        Args:
            waveform:
              Audio waveform tensor of shape `[channels, samples]`.
            sample_rate:
              Sample rate of the input audio in Hz.

        The magnitude scalogram is mean-pooled along the time axis into
        non-overlapping hop windows before log compression, yielding a
        temporally downsampled representation.

        Returns:
            Wavelet spectrogram (scalogram) array of shape
            `[n_scales, ceil(n_samples / hop_length)]`.

        Raises:
            ValueError:
              If sample rate mismatch or invalid input shape.
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

        # Take first channel (mono)
        mono_waveform = waveform[0].numpy()  # [samples,]

        # Compute FFT of the signal
        n_samples = len(mono_waveform)
        fft_signal = np.fft.rfft(mono_waveform)  # [n_freqs,]
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)

        # Compute CWT for each scale using FFT-based convolution
        cwt_matrix = np.zeros((len(self.config.widths), n_samples))

        for i, scale in enumerate(self.config.widths):
            # Wavelet spectrum at this scale: Ψ*(a·f)
            # For real-valued wavelets, the Fourier transform is also real,
            # so we just need to evaluate the wavelet's spectrum at scaled frequencies.
            wavelet_spectrum = self._get_wavelet_spectrum(freqs * scale)

            # CWT(a, b) = F^{-1}{ X(f) · Ψ*(a·f) }
            # Specify n=n_samples to handle odd-length inputs correctly.
            cwt_at_scale = np.fft.irfft(fft_signal * wavelet_spectrum, n=n_samples)
            cwt_matrix[i] = cwt_at_scale

        # Mean-pool the magnitude along the time axis before log compression
        # to match the dynamic range of mel-spectrograms.
        pooled_magnitude = self._mean_pool_time(
            magnitude=np.abs(cwt_matrix), hop_length=self.config.hop_length
        )
        log_scalogram = np.log(pooled_magnitude + 1e-8)

        if not np.all(np.isfinite(log_scalogram)):
            log_scalogram = np.nan_to_num(
                log_scalogram, nan=0.0, posinf=0.0, neginf=0.0
            )

        return log_scalogram

    def _mean_pool_time(
        self, magnitude: npt.NDArray[np.floating], hop_length: int
    ) -> npt.NDArray[np.floating]:
        """Mean-pool a magnitude scalogram along the time axis.

        Splits the time axis into non-overlapping windows of `hop_length`
        samples and averages each window. The final partial window is
        averaged over its real samples only, without zero-padding.

        Args:
            magnitude:
              Magnitude scalogram of shape `[n_scales, n_samples]`.
            hop_length:
              Number of samples per pooling window.

        Returns:
            Pooled scalogram of shape
            `[n_scales, ceil(n_samples / hop_length)]`.
        """
        n_scales, n_samples = magnitude.shape
        n_frames = int(np.ceil(n_samples / hop_length))
        pooled = np.empty((n_scales, n_frames), dtype=magnitude.dtype)
        for frame in range(n_frames):
            start = frame * hop_length
            end = min(start + hop_length, n_samples)
            pooled[:, frame] = magnitude[:, start:end].mean(axis=1)
        return pooled

    def _get_wavelet_spectrum(
        self, freqs: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Return the Fourier transform of the mother wavelet at given frequencies.

        The CWT is computed as the inverse FFT of the product between the signal's
        FFT and the wavelet's spectrum. This method provides the wavelet spectrum
        for a given set of frequencies.

        Args:
            freqs:
              Array of frequencies in Hz at which to evaluate the wavelet spectrum.

        Returns:
            Wavelet spectrum values at each frequency.

        Raises:
            ValueError: If an unsupported wavelet name is configured.
        """
        wavelet_spectrum_map: dict[str, callable] = {
            "ricker": _ricker_spectrum,
            "morl": _morl_spectrum,
            "cgau8": _cgau8_spectrum,
            "mexh": _mexh_spectrum,
        }

        if self.config.wavelet not in wavelet_spectrum_map:
            raise ValueError(
                f"Unsupported wavelet '{self.config.wavelet}'. "
                f"Valid options: {sorted(wavelet_spectrum_map.keys())}"
            )

        return wavelet_spectrum_map[self.config.wavelet](freqs)

    def get_feature_shape(self, duration_seconds: float) -> tuple[int, int]:
        """Compute the expected feature shape for a given duration.

        Args:
            duration_seconds:
              Duration of audio in seconds.

        Returns:
            Tuple of (n_scales, time_steps), where time_steps is the number
            of hop windows after temporal mean-pooling.
        """
        samples = int(duration_seconds * self.config.sample_rate)
        time_steps = int(np.ceil(samples / self.config.hop_length))
        return (self.config.n_scales, time_steps)


def extract_wavelet_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int,
    config: WaveletSpectrogramConfig | None = None,
) -> npt.NDArray[np.floating]:
    """Convenience function for extracting wavelet spectrogram features.

    Args:
        waveform:
          Audio waveform of shape `[channels, samples]`.
        sample_rate:
          Sample rate in Hz.
        config:
          Feature configuration. Uses defaults if None.

    Returns:
        Wavelet spectrogram array of shape `[n_scales, time_steps]`.

    Example:
        >>> waveform = torch.randn(1, 16000)
        >>> features = extract_wavelet_spectrogram(waveform, sample_rate=16000)
        >>> features.shape[0]  # Number of scales (default: 48)
        48
    """
    extractor = WaveletSpectrogramExtractor(config)
    return extractor.extract(waveform, sample_rate)
