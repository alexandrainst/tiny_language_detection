"""Feature extraction modules.

MFCCs are standard lightweight features for edge audio classification
(Liu, 2026). Wavelet transforms provide alternative time-frequency
representations with superior frequency localisation at low frequencies
(Fahim et al., 2025; Bin Liu et al., 2026).
"""

from .mel_spectrogram import (
    WaveletSpectrogramConfig,
    WaveletSpectrogramExtractor,
    extract_wavelet_spectrogram,
)
from .mfcc import extract_mfcc

__all__ = [
    "extract_mfcc",
    "WaveletSpectrogramConfig",
    "WaveletSpectrogramExtractor",
    "extract_wavelet_spectrogram",
]
