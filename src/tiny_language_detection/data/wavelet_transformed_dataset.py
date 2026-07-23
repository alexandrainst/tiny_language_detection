"""Dataset with on-the-fly wavelet spectrogram extraction.

Mirrors the TransformedDataset from Phase 2 but extracts wavelet features
using the Continuous Wavelet Transform (CWT) instead of Log Mel-spectrograms.

The CWT produces a time-frequency representation called a scalogram, which
captures multi-scale frequency content with better localisation than STFT-based
features at low frequencies (Fahim et al., 2025; Bin Liu et al., 2026).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from tiny_language_detection.config.defaults import Config
from tiny_language_detection.data.preprocessing import load_and_preprocess
from tiny_language_detection.features.mel_spectrogram import WaveletSpectrogramConfig
from tiny_language_detection.features.mel_spectrogram import (
    extract_wavelet_spectrogram as _extract_wavelet_spectrogram,
)

logger = logging.getLogger(__name__)


class WaveletTransformedDataset(Dataset):
    """Dataset with on-the-fly wavelet spectrogram extraction.

    Loads audio files and extracts wavelet spectrogram features using the
    Continuous Wavelet Transform (CWT). The output is a time-frequency
    representation where the frequency dimension corresponds to CWT scales
    rather than Mel bins.

    Attributes:
        df:
          Pandas DataFrame with manifest data.
        data_dir:
          Directory containing audio files.
        config:
          Audio configuration (sample rate, etc.).
        wavelet_config:
          Wavelet spectrogram extraction configuration.
        lang_to_label:
          Mapping from language codes to integer labels.
        label_to_lang:
          Reverse mapping from integer labels to language codes.
    """

    def __init__(
        self,
        manifest_path: Path,
        data_dir: Path,
        config: Config,
        wavelet_config: WaveletSpectrogramConfig | None = None,
    ) -> None:
        """Initialise the dataset.

        Args:
            manifest_path:
              Path to CSV manifest file with columns: path, language, label.
            data_dir:
              Directory containing audio files.
            config:
              Audio configuration (sample rate, etc.).
            wavelet_config:
              Wavelet spectrogram extraction configuration. Uses defaults if None.

        Raises:
            ValueError:
              If the manifest is missing any required columns.
        """
        self.df = pd.read_csv(manifest_path)
        self.data_dir = data_dir
        self.config = config
        self.wavelet_config = wavelet_config or WaveletSpectrogramConfig(
            sample_rate=config.sample_rate,
            widths=config.wavelet_n_scales,
            wavelet=config.wavelet_wavelet,
        )

        # Validate required columns
        required_columns = {"path", "language", "label"}
        if not required_columns.issubset(self.df.columns):
            raise ValueError(
                f"Expected columns: {required_columns}, got {self.df.columns.tolist()}"
            )

        # Build language to label mapping
        self.lang_to_label = {
            lang: idx for idx, lang in enumerate(sorted(self.df["language"].unique()))
        }
        self.label_to_lang = {v: k for k, v in self.lang_to_label.items()}

    def __len__(self) -> int:
        """Return number of samples.

        Returns:
            Number of rows in the manifest DataFrame.
        """
        return len(self.df)

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str, str]:
        """Get a single sample with on-the-fly wavelet extraction.

        Args:
            index:
              Sample index.

        Returns:
            Tuple of (wavelet_spec, label, language, duration_group).
        """
        row = self.df.iloc[index]
        label = int(row["label"])
        language = str(row["language"])
        duration_group = str(row.get("duration_group", "unknown"))

        # Construct audio path
        lang = str(row["language"])
        audio_filename = row["path"]

        # Handle different directory structures
        if lang == "en":
            # English: extracted to cv26-en/clips/
            audio_path = self.data_dir / "cv26-en" / "clips" / audio_filename
        else:
            # Danish: extracted to cv26-da/
            audio_path = self.data_dir / "cv26-da" / audio_filename

        # Load and preprocess audio (returns waveform at target_sr)
        waveform = load_and_preprocess(
            audio_path=audio_path, target_sr=self.config.sample_rate
        )

        # Extract wavelet spectrogram
        wavelet_spec = _extract_wavelet_spectrogram(
            waveform=waveform,
            sample_rate=self.config.sample_rate,
            config=self.wavelet_config,
        )

        # Convert to tensor and add channel dimension for CNN compatibility
        # Shape: [n_scales, time_steps] -> [1, n_scales, time_steps]
        wavelet_tensor = torch.from_numpy(wavelet_spec).float().unsqueeze(0)

        return wavelet_tensor, label, language, duration_group
