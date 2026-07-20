"""Default configuration values for the language detection pipeline.

Sample rate follows edge audio classification standards (Liu, 2026).
MFCC window/hop parameters validated for commodity edge hardware
(Darsi et al., 2026). Wavelet feature parameters follow Fahim et al. (2025)
and Bin Liu et al. (2026) for CWT-based audio classification.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Default configuration for the tiny language detection pipeline.

    Attributes:
        sample_rate:
          Audio sample rate in Hz (16 kHz standard for edge audio; Liu, 2026).
        mono:
          Whether to convert audio to mono.
        mfcc_num_coeffs:
          Number of MFCC coefficients to extract (13-40 valid range).
        mfcc_window_ms:
          Analysis window length in ms (standard STFT params; Darsi et al., 2026).
        mfcc_hop_ms:
          Hop length between analysis windows in milliseconds.
        wavelet_n_scales:
          Number of CWT scales for wavelet spectrogram extraction
          (32-64 recommended; Fahim et al., 2025).
        wavelet_wavelet:
          Mother wavelet name for CWT. Options: "ricker", "morl", "cgau8".
        languages:
          List of language codes to detect.
        label_map:
          Mapping from language codes to integer labels.
        train_test_split:
          Proportion of data to use for training.
        duration_groups:
          Duration-based grouping boundaries in seconds.
        data_dir:
          Root directory for data files.
        raw_audio_dir:
          Directory for raw audio files.
        processed_dir:
          Directory for processed feature files.
        models_dir:
          Directory for saved models.
        results_dir:
          Directory for experiment results.
    """

    # Audio settings
    sample_rate: int = 16_000
    mono: bool = True

    # MFCC feature extraction
    mfcc_num_coeffs: int = 20
    mfcc_window_ms: int = 25
    mfcc_hop_ms: int = 10

    # Wavelet feature extraction (CWT)
    wavelet_n_scales: int = 48
    wavelet_wavelet: str = "ricker"

    # Language settings
    languages: list[str] = field(default_factory=lambda: ["da", "en"])
    label_map: dict[str, int] = field(default_factory=lambda: {"da": 0, "en": 1})

    # Data splitting
    train_test_split: float = 0.8

    # Duration-based grouping (in seconds)
    duration_groups: list[tuple[int, int | None]] = field(
        default_factory=lambda: [(0, 2), (2, 4), (4, 6), (6, None)]
    )

    # Directory paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    raw_audio_dir: Path = field(default_factory=lambda: Path("data") / "raw_audio")
    processed_dir: Path = field(default_factory=lambda: Path("data") / "processed")
    models_dir: Path = field(default_factory=lambda: Path("data") / "models")
    results_dir: Path = field(default_factory=lambda: Path("results"))
