#!/usr/bin/env -S uv run
"""Train CNN-RNN model with Log Mel-spectrogram features (Phase 2).

Usage:
    uv run src/scripts/train_phase2.py --epochs 30 --batch-size 32 --lr 0.001
    uv run src/scripts/train_phase2.py --epochs 30 --batch-size 32 --lr 0.001 \
        --hidden-size 128 --num-layers 2

Hyperparameters to tune:
    --n-mels: Number of Mel bins (64, 80, 128)
    --hidden-size: GRU hidden dimension (64, 128)
    --num-layers: Number of GRU layers (1, 2)
    --lr: Learning rate (0.001, 0.0005, 0.0001)
    --batch-size: Batch size (32, 64)
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from tiny_language_detection.config import Config
from tiny_language_detection.data.preprocessing import load_and_preprocess
from tiny_language_detection.features.mel_spectrogram import (
    MelSpectrogramConfig,
    extract_log_mel_spectrogram,
)
from tiny_language_detection.models import CNNRNNLanguageDetector


def setup_logging() -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


logger = logging.getLogger(__name__)


class TransformedDataset(Dataset):
    """Dataset with on-the-fly Log Mel-spectrogram extraction.

    Similar to Phase 1 but extracts Log Mel-spectrograms instead of MFCCs.
    """

    def __init__(
        self,
        manifest_path: Path,
        data_dir: Path,
        config: Config,
        mel_config: MelSpectrogramConfig,
    ) -> None:
        """Initialise the dataset.

        Args:
            manifest_path: Path to CSV manifest file.
            data_dir: Directory containing audio files.
            config: Audio configuration.
            mel_config: Mel-spectrogram configuration.

        Raises:
            ValueError:
                If the manifest is missing any required columns.
        """
        self.df = pd.read_csv(manifest_path)
        self.data_dir = data_dir
        self.config = config
        self.mel_config = mel_config

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
        """Return number of samples."""
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str, str]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (log_mel_spec, label, language, duration_group).
        """
        row = self.df.iloc[idx]
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

        # Extract Log Mel-spectrogram
        log_mel_spec = extract_log_mel_spectrogram(
            waveform=waveform,
            sample_rate=self.config.sample_rate,
            config=self.mel_config,
        )

        # Convert to tensor and add channel dimension
        log_mel_tensor = torch.from_numpy(log_mel_spec).float()

        return log_mel_tensor, label, language, duration_group


def collate_fn(
    batch: list[tuple[torch.Tensor, int, str, str]],
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    """Collate function for variable-length sequences.

    Pads Log Mel-spectrograms to the maximum time steps in the batch.

    Args:
        batch: List of (log_mel_spec, label, language, duration_group) tuples.

    Returns:
        Tuple of (padded_specs, labels, languages, duration_groups).
    """
    specs, labels, languages, duration_groups = zip(*batch)

    # Find max time steps
    max_time = max(spec.size(1) for spec in specs)

    # Pad sequences
    padded_specs = []
    for spec in specs:
        # spec shape: [n_mels, time]
        padding = torch.zeros(spec.size(0), max_time - spec.size(1))
        padded = torch.cat([spec, padding], dim=1)
        padded_specs.append(padded)

    specs_tensor = torch.stack(padded_specs)  # [batch, n_mels, max_time]
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return specs_tensor, labels_tensor, list(languages), list(duration_groups)


class TrainRecord(TypedDict):
    """Training record for a single epoch."""

    epoch: int
    train_loss: float
    train_accuracy: float


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch.

    Args:
        model: The neural network model.
        dataloader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.

    Returns:
        Tuple of (average_loss, accuracy) for the epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (specs, labels, _, _) in enumerate(dataloader):
        specs = specs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def main() -> None:
    """Main training entry point.

    Raises:
        FileNotFoundError:
            If the training manifest does not exist.
    """
    parser = argparse.ArgumentParser(
        description="Train CNN-RNN model for language detection (Phase 2)"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs (default: 30)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--n-mels", type=int, default=80, help="Number of Mel bins (default: 80)"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=64, help="GRU hidden size (default: 64)"
    )
    parser.add_argument(
        "--num-layers", type=int, default=1, help="Number of GRU layers (default: 1)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experiments/phase2"),
        help="Output directory (default: data/experiments/phase2)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=2,
        help="Phase number for compatibility (default: 2)",
    )

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    config = Config()
    mel_config = MelSpectrogramConfig(
        sample_rate=config.sample_rate, n_mels=args.n_mels, n_fft=400, hop_length=160
    )

    logger.info("Phase 2: Log Mel-spectrogram + CNN-RNN")
    logger.info(
        f"Mel bins: {args.n_mels}, Hidden size: {args.hidden_size}, "
        f"Layers: {args.num_layers}"
    )
    logger.info(
        f"Batch size: {args.batch_size}, Learning rate: {args.lr}, "
        f"Epochs: {args.epochs}"
    )

    # Load dataset
    train_manifest = args.data_dir / "sampled" / "train.csv"
    if not train_manifest.exists():
        raise FileNotFoundError(f"Training manifest not found: {train_manifest}")

    logger.info(f"Loading training data from {train_manifest}")
    dataset = TransformedDataset(
        manifest_path=train_manifest,
        data_dir=args.data_dir,
        config=config,
        mel_config=mel_config,
    )

    logger.info(f"Training samples: {len(dataset)}")

    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Model
    num_languages = len(dataset.lang_to_label)
    model = CNNRNNLanguageDetector(
        n_mels=args.n_mels,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_languages=num_languages,
    ).to(device)

    total_params = model.count_parameters()
    logger.info(f"Model has {total_params:,} trainable parameters")

    # Save label mapping
    label_map_path = args.output_dir / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(dataset.label_to_lang, f, indent=2)
    logger.info(f"Saved label map to {label_map_path}")

    # Compute class weights for imbalanced data
    label_counts = Counter([s[1] for s in [dataset[i] for i in range(len(dataset))]])
    total = sum(label_counts.values())
    class_weights = [
        total / (num_languages * label_counts.get(i, 1)) for i in range(num_languages)
    ]
    logger.info(f"Class weights: {class_weights}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    history: list[TrainRecord] = []

    logger.info("Starting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_acc, 2),
        }
        history.append(record)

        logger.info(
            f"Epoch {epoch}/{args.epochs} - Loss: {train_loss:.6f} - "
            f"Accuracy: {train_acc:.2f}%"
        )

    # Save outputs
    model_path = args.output_dir / "model.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model weights to {model_path}")

    config_path = args.output_dir / "config.json"
    config_dict = {
        "n_mels": args.n_mels,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "sample_rate": config.sample_rate,
    }
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Saved training config to {config_path}")

    history_path = args.output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training history to {history_path}")

    logger.info("============================================================")
    logger.info("Training complete!")
    logger.info(f"Final accuracy: {history[-1]['train_accuracy']:.2f}%")
    logger.info("============================================================")


if __name__ == "__main__":
    main()
