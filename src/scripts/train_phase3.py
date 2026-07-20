"""Training script for wavelet spectrogram + CNN model (Phase 3).

Uses the Continuous Wavelet Transform (CWT) to extract time-frequency features,
then trains a lightweight CNN for binary language detection (Danish vs English).

Wavelet features provide superior frequency localisation at low frequencies
compared to STFT-based features, as reported by Fahim et al. (2025) and
Bin Liu et al. (2026).

Example:
    uv run src/scripts/train_phase3.py --phase 3 \
        --wavelet-wavelet ricker --wavelet-n-scales 48 \
        --epochs 30 --batch-size 32 --lr 0.001
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tiny_language_detection.config.defaults import Config
from tiny_language_detection.data.wavelet_transformed_dataset import (
    WaveletTransformedDataset,
)
from tiny_language_detection.features.mel_spectrogram import WaveletSpectrogramConfig
from tiny_language_detection.models.cnn import LanguageDetectionCNN

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging for the training script."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


@dataclass
class TrainRecord:
    """Training record for a single epoch.

    Attributes:
        epoch:
          Epoch number (1-indexed).
        train_loss:
          Average training loss for the epoch.
        train_accuracy:
          Training accuracy percentage for the epoch.
    """

    epoch: int
    train_loss: float
    train_accuracy: float


def collate_fn(
    batch: list[tuple[torch.Tensor, int, str, str]],
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    """Collate function for variable-length wavelet sequences.

    Pads wavelet spectrograms to the maximum time steps in the batch.

    Args:
        batch:
          List of (wavelet_spec, label, language, duration_group) tuples.

    Returns:
        Tuple of (padded_specs, labels, languages, duration_groups).
    """
    specs, labels, languages, duration_groups = zip(*batch)

    # Find max time steps across the batch
    max_time = max(spec.size(2) for spec in specs)

    # Pad sequences along the time dimension
    padded_specs = []
    for spec in specs:
        # spec shape: [1, n_scales, time]
        padding = torch.zeros(1, spec.size(1), max_time - spec.size(2))
        padded = torch.cat([spec, padding], dim=2)
        padded_specs.append(padded)

    specs_tensor = torch.stack(padded_specs)  # [batch, 1, n_scales, max_time]
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return specs_tensor, labels_tensor, list(languages), list(duration_groups)


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch.

    Args:
        model:
          The neural network model.
        dataloader:
          Training data loader.
        criterion:
          Loss function.
        optimizer:
          Optimizer.
        device:
          Device to train on.

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
        description=(
            "Train CNN model for language detection with wavelet features (Phase 3)"
        )
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
        "--wavelet-wavelet",
        type=str,
        default="ricker",
        help="Mother wavelet name (default: ricker)",
    )
    parser.add_argument(
        "--wavelet-n-scales",
        type=int,
        default=48,
        help="Number of CWT scales (default: 48)",
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
        default=Path("data/experiments/phase3"),
        help="Output directory (default: data/experiments/phase3)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=3,
        help="Phase number for compatibility (default: 3)",
    )

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    config = Config()
    wavelet_config = WaveletSpectrogramConfig(
        sample_rate=config.sample_rate,
        widths=args.wavelet_n_scales,
        wavelet=args.wavelet_wavelet,
    )

    logger.info("Phase 3: Wavelet Spectrogram + CNN")
    logger.info(f"CWT wavelet: {args.wavelet_wavelet}, Scales: {args.wavelet_n_scales}")
    logger.info(
        f"Batch size: {args.batch_size}, Learning rate: {args.lr}, "
        f"Epochs: {args.epochs}"
    )

    # Load dataset
    train_manifest = args.data_dir / "sampled" / "train.csv"
    if not train_manifest.exists():
        raise FileNotFoundError(f"Training manifest not found: {train_manifest}")

    logger.info(f"Loading training data from {train_manifest}")
    dataset = WaveletTransformedDataset(
        manifest_path=train_manifest,
        data_dir=args.data_dir,
        config=config,
        wavelet_config=wavelet_config,
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
    model = LanguageDetectionCNN(
        num_mfcc=wavelet_config.n_scales,
        time_steps=50,  # Will be overridden by adaptive pooling
        num_languages=num_languages,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
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
        "wavelet": args.wavelet_wavelet,
        "n_scales": args.wavelet_n_scales,
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
        json.dump(asdict(history), f, indent=2)
    logger.info(f"Saved training history to {history_path}")

    logger.info("============================================================")
    logger.info("Training complete!")
    logger.info(f"Final accuracy: {history[-1]['train_accuracy']:.2f}%")
    logger.info("============================================================")


if __name__ == "__main__":
    setup_logging()
    main()
