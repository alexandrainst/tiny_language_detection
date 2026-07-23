"""Multi-class language detection dataset."""

import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset

from datasets import load_dataset
from tiny_language_detection.data.preprocessing import load_and_preprocess
from tiny_language_detection.features.mel_spectrogram import (
    MelSpectrogramConfig,
    extract_log_mel_spectrogram,
)
from tiny_language_detection.features.spec_augment import SpecAugment

logger = logging.getLogger(__name__)


class MulticlassDataset(Dataset):
    """Dataset for multi-class language detection.

    Supports two modes:
    - HF: Load from HuggingFace datasets (e.g., YODAS-Granary)
    - Manifest: Load from CSV manifest with audio paths
    """

    def __init__(
        self,
        source: str,
        mel_config: MelSpectrogramConfig,
        time_mask_param: int = 5,
        freq_mask_param: int = 4,
        use_augment: bool = True,
        use_hf: bool = False,
        hf_split: str = "train",
        data_dir: Path | None = None,
        time_masks: int = 1,
        freq_masks: int = 1,
    ) -> None:
        """Initialise the dataset.

        Args:
            source:
                HF dataset name or path to manifest CSV.
            mel_config:
                Mel spectrogram configuration.
            time_mask_param:
                Time mask length. Defaults to 5.
            freq_mask_param:
                Frequency mask length. Defaults to 4.
            use_augment:
                Whether to apply SpecAugment. Defaults to True.
            use_hf:
                Load from HuggingFace datasets. Defaults to False.
            hf_split:
                Dataset split to load. Defaults to "train".
            data_dir:
                Base directory for audio files. Defaults to None.
            time_masks:
                Number of time masks. Defaults to 1.
            freq_masks:
                Number of frequency masks. Defaults to 1.
        """
        self.samples: list = []
        self.labels = []
        self.languages = []
        self.mel_config = mel_config
        self.use_augment = use_augment
        self.use_hf = use_hf
        self.data_dir = data_dir or Path("data")

        self.augment = SpecAugment(
            time_mask_param=time_mask_param,
            freq_mask_param=freq_mask_param,
            time_masks=time_masks,
            freq_masks=freq_masks,
        )

        if use_augment:
            logger.info(
                f"SpecAugment: time_mask={time_mask_param} (×{time_masks}), "
                f"freq_mask={freq_mask_param} (×{freq_masks})"
            )

        # First pass: collect all unique languages from dataset
        temp_languages = []

        if use_hf:
            # Load from HuggingFace datasets
            logger.info(f"Loading HF dataset: {source}, split={hf_split}")

            ds = load_dataset(source, split=hf_split, streaming=False)

            for sample in ds:
                lang = sample["lang"]
                temp_languages.append(lang)
                self.samples.append(sample["audio"])
                self.languages.append(lang)

            logger.info(f"Loaded {len(self.samples)} samples from HF")
        else:
            # Load from CSV manifest
            manifest_path = Path(source)
            logger.info(f"Loading manifest: {manifest_path}")

            with open(manifest_path, "r") as f:
                lines = f.readlines()[1:]  # Skip header

            for line in lines:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    audio_filename = parts[0]
                    language = parts[1]
                    temp_languages.append(language)
                    self.samples.append((audio_filename, language))
                    self.languages.append(language)

            logger.info(f"Loaded {len(self.samples)} samples from manifest")

        # Build language-to-index mapping from actual data
        unique_langs = sorted(set(temp_languages))
        lang_to_id = {lang: i for i, lang in enumerate(unique_langs)}
        self.lang_to_id = lang_to_id
        self.labels = [lang_to_id[lang] for lang in self.languages]

        logger.info(
            f"Languages ({len(unique_langs)}): {', '.join(unique_langs[:10])}"
            + (
                f" ... ({len(unique_langs) - 10} more)"
                if len(unique_langs) > 10
                else ""
            )
        )

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        """Get a single sample."""
        if self.use_hf:
            audio_dict = self.samples[idx]
            waveform = audio_dict["array"]  # type: ignore[index]
            sr = audio_dict.get("sampling_rate", 16000)  # type: ignore[union-attr]
        else:
            audio_filename, language = self.samples[idx]
            audio_path = self._get_audio_path(audio_filename, language)
            waveform = load_and_preprocess(str(audio_path), target_sr=16000)
            sr = 16000

        # Extract features
        spec = extract_log_mel_spectrogram(waveform, sr, self.mel_config)
        spec_tensor = torch.from_numpy(spec).float()

        # Apply SpecAugment (training only)
        if self.use_augment:
            spec_tensor = self.augment(spec_tensor.unsqueeze(0)).squeeze(0)

        label = self.labels[idx]
        return spec_tensor, label, self.languages[idx]

    def _get_audio_path(self, filename: str, language: str) -> Path:
        """Get audio file path for manifest-based loading.

        Expects data directory structure:
        {data_dir}/
          da/
            audio1.wav
          en/
            audio2.wav

        Returns:
            Path to audio file.
        """
        return self.data_dir / language / filename


def collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function for variable-length spectrograms.

    Args:
        batch: List of (spectrogram, label, language_name) tuples.

    Returns:
        Tuple of (padded_spectrograms, labels) tensors.
    """
    specs, labels, _ = zip(*batch)
    max_time = max(s.shape[1] for s in specs)
    padded_specs = []
    for spec in specs:
        pad = torch.zeros(spec.shape[0], max_time - spec.shape[1])
        padded = torch.cat([spec, pad], dim=1)
        padded_specs.append(padded)
    return torch.stack(padded_specs).unsqueeze(1), torch.tensor(
        labels, dtype=torch.long
    )
