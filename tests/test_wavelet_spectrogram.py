"""Tests for wavelet spectrogram feature extraction.

Wavelet transforms provide alternative time-frequency representations
with superior frequency localisation at low frequencies (Fahim et al., 2025;
Bin Liu et al., 2026).
"""

import numpy as np
import pytest
import torch

from src.tiny_language_detection.features.mel_spectrogram import (
    WaveletSpectrogramConfig,
    WaveletSpectrogramExtractor,
    extract_wavelet_spectrogram,
)

# Test constants
SAMPLE_RATE = 16000
DURATION_1S = 1.0
DURATION_2S = 2.0
SAMPLES_1S = int(SAMPLE_RATE * DURATION_1S)
SAMPLES_2S = int(SAMPLE_RATE * DURATION_2S)

# Generate synthetic sine wave for testing (440 Hz, A note)


def _make_sine_wave(
    duration_seconds: float, frequency_hz: float, sample_rate: int
) -> torch.Tensor:
    """Generate a synthetic sine wave waveform.

    Args:
        duration_seconds: Duration of the waveform in seconds.
        frequency_hz: Frequency of the sine wave in Hz.
        sample_rate: Sample rate in Hz.

    Returns:
        Waveform tensor of shape [1, samples].
    """
    n_samples = int(sample_rate * duration_seconds)
    t = torch.tensor(np.linspace(0, duration_seconds, n_samples, endpoint=False))
    waveform = (torch.sin(t * 2 * np.pi * frequency_hz)).unsqueeze(0)
    return waveform


# --- WaveletSpectrogramConfig tests ---


class TestWaveletSpectrogramConfig:
    """Tests for WaveletSpectrogramConfig."""

    def test_default_config(self) -> None:
        """Default configuration uses sensible defaults."""
        config = WaveletSpectrogramConfig()
        assert config.sample_rate == 16000
        assert config.wavelet == "ricker"
        assert config.f_min == 0.0
        assert config.f_max == SAMPLE_RATE / 2

    def test_custom_widths(self) -> None:
        """Custom number of scales is applied."""
        config = WaveletSpectrogramConfig(widths=64, sample_rate=SAMPLE_RATE)
        assert len(config.widths) == 64
        assert config.n_scales == 64

    def test_custom_wavelet(self) -> None:
        """Custom wavelet name is stored."""
        config = WaveletSpectrogramConfig(wavelet="morl", sample_rate=SAMPLE_RATE)
        assert config.wavelet == "morl"

    def test_custom_f_max(self) -> None:
        """Custom f_max overrides Nyquist default."""
        config = WaveletSpectrogramConfig(f_max=4000.0, sample_rate=SAMPLE_RATE)
        assert config.f_max == 4000.0

    def test_invalid_wavelet_raises(self) -> None:
        """Invalid wavelet name raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported wavelet"):
            WaveletSpectrogramConfig(wavelet="invalid", sample_rate=SAMPLE_RATE)

    def test_negative_widths_raises(self) -> None:
        """Negative number of widths raises ValueError."""
        with pytest.raises(ValueError, match="widths must be a positive integer"):
            WaveletSpectrogramConfig(widths=-1, sample_rate=SAMPLE_RATE)

    def test_zero_widths_raises(self) -> None:
        """Zero number of widths raises ValueError."""
        with pytest.raises(ValueError, match="widths must be a positive integer"):
            WaveletSpectrogramConfig(widths=0, sample_rate=SAMPLE_RATE)

    def test_log_spaced_widths(self) -> None:
        """Widths are logarithmically spaced for uniform frequency coverage."""
        config = WaveletSpectrogramConfig(widths=32, sample_rate=SAMPLE_RATE)
        # Check that widths increase (log-spaced from f_min to f_max)
        assert np.all(np.diff(config.widths) > 0), (
            "Widths should be monotonically increasing"
        )

    def test_width_array_input(self) -> None:
        """Pre-computed width array is accepted."""
        custom_widths = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
        config = WaveletSpectrogramConfig(widths=custom_widths, sample_rate=SAMPLE_RATE)
        assert len(config.widths) == 5
        assert np.allclose(config.widths, custom_widths.astype(np.float64))

    def test_empty_width_array_raises(self) -> None:
        """Empty width array raises ValueError."""
        with pytest.raises(ValueError, match="widths array must not be empty"):
            WaveletSpectrogramConfig(widths=np.array([]), sample_rate=SAMPLE_RATE)

    def test_2d_width_array_raises(self) -> None:
        """2D width array raises ValueError."""
        with pytest.raises(ValueError, match="widths must be a 1D array"):
            WaveletSpectrogramConfig(
                widths=np.array([[1.0, 2.0], [3.0, 4.0]]), sample_rate=SAMPLE_RATE
            )

    def test_invalid_width_type_raises(self) -> None:
        """Invalid width type raises ValueError."""
        with pytest.raises(ValueError, match="widths must be int or ndarray"):
            WaveletSpectrogramConfig(widths="invalid", sample_rate=SAMPLE_RATE)  # noqa: S105

    def test_n_scales_property(self) -> None:
        """n_scales property returns correct count."""
        config = WaveletSpectrogramConfig(widths=48, sample_rate=SAMPLE_RATE)
        assert config.n_scales == 48


# --- WaveletSpectrogramExtractor tests ---


class TestWaveletSpectrogramExtractor:
    """Tests for WaveletSpectrogramExtractor."""

    def test_extract_shape_1s(self) -> None:
        """Extracted features have correct shape for 1-second audio."""
        extractor = WaveletSpectrogramExtractor(
            config=WaveletSpectrogramConfig(sample_rate=SAMPLE_RATE, widths=48)
        )
        waveform = _make_sine_wave(DURATION_1S, 440.0, SAMPLE_RATE)
        features = extractor.extract(waveform, sample_rate=SAMPLE_RATE)
        assert features.shape == (48, SAMPLES_1S)

    def test_extract_shape_2s(self) -> None:
        """Extracted features scale with duration."""
        extractor = WaveletSpectrogramExtractor(
            config=WaveletSpectrogramConfig(sample_rate=SAMPLE_RATE, widths=32)
        )
        waveform = _make_sine_wave(DURATION_2S, 440.0, SAMPLE_RATE)
        features = extractor.extract(waveform, sample_rate=SAMPLE_RATE)
        assert features.shape == (32, SAMPLES_2S)

    def test_extract_with_custom_widths(self) -> None:
        """Custom number of scales produces correct shape."""
        extractor = WaveletSpectrogramExtractor(
            config=WaveletSpectrogramConfig(sample_rate=SAMPLE_RATE, widths=64)
        )
        waveform = _make_sine_wave(DURATION_1S, 440.0, SAMPLE_RATE)
        features = extractor.extract(waveform, sample_rate=SAMPLE_RATE)
        assert features.shape[0] == 64

    def test_extract_mismatched_sample_rate_raises(self) -> None:
        """Sample rate mismatch raises ValueError."""
        extractor = WaveletSpectrogramExtractor(
            config=WaveletSpectrogramConfig(sample_rate=SAMPLE_RATE)
        )
        waveform = _make_sine_wave(DURATION_1S, 440.0, SAMPLE_RATE)
        with pytest.raises(ValueError, match="Expected sample rate"):
            extractor.extract(waveform, sample_rate=8000)

    def test_extract_multidimensional_raises(self) -> None:
        """Non-2D waveform raises ValueError."""
        extractor = WaveletSpectrogramExtractor(
            config=WaveletSpectrogramConfig(sample_rate=SAMPLE_RATE)
        )
        # 3D tensor should raise
        waveform_3d = torch.randn(1, 160, SAMPLES_1S)
        with pytest.raises(ValueError, match="Expected 2D"):
            extractor.extract(waveform_3d, sample_rate=SAMPLE_RATE)

    def test_extract_returns_finite_values(self) -> None:
        """Extracted features contain finite values (no NaN/Inf)."""
        extractor = WaveletSpectrogramExtractor(
            config=WaveletSpectrogramConfig(sample_rate=SAMPLE_RATE, widths=48)
        )
        waveform = _make_sine_wave(DURATION_1S, 440.0, SAMPLE_RATE)
        features = extractor.extract(waveform, sample_rate=SAMPLE_RATE)
        assert np.all(np.isfinite(features)), (
            "Features should contain no NaN or Inf values"
        )

    def test_extract_returns_negative_values(self) -> None:
        """Log-compressed features can be negative (log of values < 1)."""
        extractor = WaveletSpectrogramExtractor(
            config=WaveletSpectrogramConfig(sample_rate=SAMPLE_RATE, widths=48)
        )
        waveform = _make_sine_wave(DURATION_1S, 440.0, SAMPLE_RATE)
        features = extractor.extract(waveform, sample_rate=SAMPLE_RATE)
        # Log compression means many values will be negative
        assert np.any(features < 0), (
            "Features should contain negative values from log compression"
        )

    def test_extract_with_different_wavelets(self) -> None:
        """Different wavelet types produce valid output."""
        for wavelet_name in ("ricker", "morl", "cgau8"):
            extractor = WaveletSpectrogramExtractor(
                config=WaveletSpectrogramConfig(
                    sample_rate=SAMPLE_RATE, widths=32, wavelet=wavelet_name
                )
            )
            waveform = _make_sine_wave(DURATION_1S, 440.0, SAMPLE_RATE)
            features = extractor.extract(waveform, sample_rate=SAMPLE_RATE)
            assert features.shape == (32, SAMPLES_1S)
            assert np.all(np.isfinite(features))

    def test_get_feature_shape(self) -> None:
        """Feature shape computation matches actual extraction."""
        extractor = WaveletSpectrogramExtractor(
            config=WaveletSpectrogramConfig(sample_rate=SAMPLE_RATE, widths=48)
        )
        expected_shape = extractor.get_feature_shape(DURATION_1S)
        assert expected_shape == (48, SAMPLES_1S)

    def test_get_feature_shape_2s(self) -> None:
        """Feature shape scales correctly with duration."""
        extractor = WaveletSpectrogramExtractor(
            config=WaveletSpectrogramConfig(sample_rate=SAMPLE_RATE, widths=32)
        )
        expected_shape = extractor.get_feature_shape(DURATION_2S)
        assert expected_shape == (32, SAMPLES_2S)


# --- Convenience function tests ---


class TestExtractWaveletSpectrogram:
    """Tests for the convenience function extract_wavelet_spectrogram."""

    def test_convenience_function(self) -> None:
        """Convenience function produces correct output shape."""
        waveform = _make_sine_wave(DURATION_1S, 440.0, SAMPLE_RATE)
        features = extract_wavelet_spectrogram(waveform, sample_rate=SAMPLE_RATE)
        assert features.shape == (48, SAMPLES_1S)

    def test_convenience_function_with_config(self) -> None:
        """Convenience function respects custom configuration."""
        config = WaveletSpectrogramConfig(sample_rate=SAMPLE_RATE, widths=64)
        waveform = _make_sine_wave(DURATION_1S, 440.0, SAMPLE_RATE)
        features = extract_wavelet_spectrogram(
            waveform, sample_rate=SAMPLE_RATE, config=config
        )
        assert features.shape == (64, SAMPLES_1S)
