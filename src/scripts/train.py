#!/usr/bin/env -S uv run
"""Training script for Phase 1: MFCC + CNN baseline.

Trains a lightweight CNN on MFCC features for Danish vs English language detection.

Usage:
    uv run src/scripts/train.py --phase 1
        [--epochs 10] [--batch-size 32] [--lr 0.001] [--seed 42]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tiny_language_detection.config import Config
from tiny_language_detection.data.preprocessing import load_and_preprocess
from tiny_language_detection.features.mfcc import extract_mfcc
from tiny_language_detection.models import LanguageDetectionCNN

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LanguageDetectionDataset(Dataset):
    """PyTorch Dataset for language detection from audio files.

    Loads audio, extracts MFCC features, and returns (features, label) pairs.
    """

    def __init__(
        self, manifest_path: Path, config: Config, data_dir: Path | None = None
    ) -> None:
        """Initialise the dataset.

        Args:
            manifest_path:
                Path to the train.csv manifest file.
            config:
                Configuration object with audio and MFCC parameters.
            data_dir (optional):
                Base directory for audio files. If None, inferred from manifest path.

        Raises:
            FileNotFoundError:
                If the manifest file does not exist.
            ValueError:
                If the manifest is missing required columns.
        """
        self.config = config
        self.samples: list[dict[str, str | int]] = []

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Train manifest not found: {manifest_path}. "
                "Please run the sampling script first to create train.csv."
            )

        # Infer data_dir from manifest path if not provided
        if data_dir is None:
            data_dir = manifest_path.parent.parent  # data/sampled -> data

        self.data_dir = Path(data_dir)

        # Load manifest
        df = pd.read_csv(manifest_path)

        if "path" not in df.columns or "language" not in df.columns:
            raise ValueError(
                "Train manifest must contain 'path' and 'language' columns."
            )

        # Build samples list
        languages_present = set(df["language"].unique())
        logger.info(f"Languages in training data: {languages_present}")

        # Warn if English is missing
        if "en" not in languages_present:
            logger.warning(
                "English data not found. Training with Danish only - "
                "model will only learn Danish characteristics."
            )

        for _, row in df.iterrows():
            lang = row["language"]
            # Construct full path based on language
            # English: data/cv26-en/clips/{clip}.mp3
            # Danish: data/cv26-da/{clip}.mp3 (extracted directly)
            audio_dir = self.data_dir / f"cv26-{lang}"
            if lang == "en":
                audio_dir = audio_dir / "clips"
            clip_path = str(row["path"])
            self.samples.append(
                {
                    "path": clip_path,
                    "audio_dir": audio_dir,
                    "language": lang,
                    "label": config.label_map.get(lang, -1),
                }
            )

        logger.info(f"Loaded {len(self.samples)} samples from manifest")

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Load a single sample.

        Args:
            idx:
                Index of the sample to load.

        Returns:
            Tuple of (mfcc_features, language_label).
        """
        sample = self.samples[idx]
        # Construct full path: audio_dir / clip_name
        audio_dir = Path(sample["audio_dir"])
        clip_name = str(sample["path"])
        audio_path = audio_dir / clip_name
        label = int(sample["label"])

        # Load and preprocess audio
        waveform = load_and_preprocess(
            audio_path=audio_path, target_sr=self.config.sample_rate
        )

        # Extract MFCC features
        mfccs = extract_mfcc(
            waveform=waveform, sample_rate=self.config.sample_rate, config=self.config
        )

        return mfccs, label


def collate_fn(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Collate function for variable-length MFCC sequences.

    Pads MFCC sequences to the maximum length in the batch.

    Args:
        batch:
            List of (mfcc_features, label) tuples.

    Returns:
        Tuple of (padded_features, lengths, labels) where:
            - padded_features: [batch_size, num_mfcc, max_time_steps]
            - lengths: list of actual sequence lengths
            - labels: [batch_size]
    """
    mfccs_list = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)

    # Get max time steps
    max_time = max(mfcc.shape[1] for mfcc in mfccs_list)

    # Pad to max length
    padded_mfccs = []
    lengths = []
    for mfcc in mfccs_list:
        time_steps = mfcc.shape[1]
        lengths.append(time_steps)

        if time_steps < max_time:
            # Pad on the right (time dimension)
            pad_size = max_time - time_steps
            padded = torch.nn.functional.pad(
                mfcc, (0, pad_size), mode="constant", value=0
            )
        else:
            padded = mfcc

        padded_mfccs.append(padded)

    # Stack: [batch_size, num_mfcc, max_time]
    padded_tensor = torch.stack(padded_mfccs, dim=0)

    return padded_tensor, labels


def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute classification accuracy.

    Args:
        outputs:
            Model logits of shape [batch_size, num_classes].
        targets:
            Ground truth labels of shape [batch_size].

    Returns:
        Accuracy as a float between 0 and 1.
    """
    predictions = outputs.argmax(dim=1)
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return correct / total if total > 0 else 0.0


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch.

    Args:
        model:
            The CNN model to train.
        dataloader:
            DataLoader providing training batches.
        criterion:
            Loss function (CrossEntropyLoss).
        optimizer:
            Optimizer (Adam).
        device:
            Device to train on (cuda/cpu).

    Returns:
        Tuple of (average_loss, accuracy) for the epoch.
    """
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    for mfccs, labels in dataloader:
        # Move to device
        mfccs = mfccs.to(device)
        labels = labels.to(device)

        # Add channel dimension: [batch, num_mfcc, time] -> [batch, num_mfcc, time, 1]
        mfccs = mfccs.unsqueeze(-1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(mfccs)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_accuracy += compute_accuracy(outputs, labels)
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return avg_loss, avg_accuracy


def save_checkpoint(
    model: nn.Module,
    config: dict,
    history: list[dict],
    label_map: dict[str, int],
    output_dir: Path,
) -> None:
    """Save model weights and training metadata.

    Args:
        model:
            Trained model to save.
        config:
            Training configuration dictionary.
        history:
            List of per-epoch loss/accuracy records.
        label_map:
            Mapping from language codes to label indices.
        output_dir:
            Directory to save outputs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    model_path = output_dir / "model.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model weights to {model_path}")

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    logger.info(f"Saved training config to {config_path}")

    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training history to {history_path}")

    # Save label map
    label_map_path = output_dir / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    logger.info(f"Saved label map to {label_map_path}")


def main() -> None:
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train MFCC+CNN model for language detection"
    )
    parser.add_argument(
        "--phase",
        type=int,
        required=True,
        help="Phase number (e.g., 1 for MFCC+CNN baseline)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)

    # Load config
    config = Config()

    # Determine paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    manifest_path = project_root / "data" / "sampled" / "train.csv"
    output_dir = project_root / "data" / "experiments" / f"phase{args.phase}"

    logger.info("=" * 60)
    logger.info("Phase %d Training", args.phase)
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Manifest: {manifest_path}")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 60)

    # Check manifest exists
    if not manifest_path.exists():
        logger.error(
            "Training manifest not found at %s. "
            "Please run the data sampling script first to create train.csv.",
            manifest_path,
        )
        sys.exit(1)

    # Create dataset
    try:
        dataset = LanguageDetectionDataset(manifest_path=manifest_path, config=config)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except ValueError as e:
        logger.error("Invalid manifest format: %s", e)
        sys.exit(1)

    if len(dataset) == 0:
        logger.error("No samples found in training manifest")
        sys.exit(1)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Determine input dimensions from first sample
    sample_mfccs, _ = dataset[0]
    num_mfcc = sample_mfccs.shape[0]
    time_steps = sample_mfccs.shape[1]
    num_languages = len(config.label_map)

    logger.info(f"Input shape: [num_mfcc={num_mfcc}, time_steps={time_steps}]")
    logger.info(f"Number of languages: {num_languages}")

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = LanguageDetectionCNN(
        num_mfcc=num_mfcc, time_steps=time_steps, num_languages=num_languages
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {total_params:,} trainable parameters")

    # Compute class weights for imbalanced data
    from collections import Counter

    label_counts = Counter([s["label"] for s in dataset.samples])
    total = sum(label_counts.values())
    # Weight = total / (num_classes * count)
    class_weights = [
        total / (num_languages * label_counts.get(i, 1)) for i in range(num_languages)
    ]
    logger.info(f"Class weights: {class_weights}")
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    history: list[dict] = []

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
            "train_accuracy": round(train_acc, 6),
        }
        history.append(record)

        logger.info(
            "Epoch %d/%d - Loss: %.6f - Accuracy: %.2f%%",
            epoch,
            args.epochs,
            train_loss,
            train_acc * 100,
        )

    # Save outputs
    training_config = {
        "phase": args.phase,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "num_mfcc": num_mfcc,
        "num_languages": num_languages,
        "model_params": total_params,
        "final_loss": history[-1]["train_loss"],
        "final_accuracy": history[-1]["train_accuracy"],
    }

    save_checkpoint(
        model=model,
        config=training_config,
        history=history,
        label_map=config.label_map,
        output_dir=output_dir,
    )

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("Final accuracy: %.2f%%", history[-1]["train_accuracy"] * 100)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
