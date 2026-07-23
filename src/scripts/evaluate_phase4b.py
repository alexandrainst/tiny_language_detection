#!/usr/bin/env python3
r"""Evaluate Phase 4b compact CNN models.

Usage:
    uv run src/scripts/evaluate_phase4b.py \\
        --checkpoint data/experiments/phase4b/tiny_cnn_kd/model_best.pth

Evaluates on the standard test set (1,729 samples) and reports:
- Overall accuracy
- Per-language accuracy (Danish, English)
- Accuracy by duration group
- Confusion matrix
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tiny_language_detection.data.preprocessing import load_and_preprocess
from tiny_language_detection.features.mel_spectrogram import (
    MelSpectrogramConfig,
    extract_log_mel_spectrogram,
)
from tiny_language_detection.models.tiny_cnn import (
    create_medium_cnn,
    create_small_cnn,
    create_tiny_cnn,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Phase4BDataset(Dataset):
    """Dataset for evaluating Phase 4b models."""

    def __init__(self, csv_path: Path, n_mels: int = 80) -> None:
        """Initialise the dataset.

        Args:
            csv_path: Path to the CSV file with test samples.
            n_mels: Number of Mel bins for feature extraction.
        """
        self.samples: list[str] = []
        self.labels: list[int] = []
        self.durations: list[float] = []
        self.mel_config = MelSpectrogramConfig(n_mels=n_mels)

        with open(csv_path, "r") as f:
            lines = f.readlines()[1:]  # Skip header

        for line in lines:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                audio_filename = parts[0]
                label = int(parts[2])  # 0=da, 1=en
                duration = float(parts[4]) if len(parts) >= 5 else None

                self.samples.append(audio_filename)
                self.labels.append(label)
                self.durations.append(duration or 0.0)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, float]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (spectrogram, label_id, duration).
        """
        audio_filename = self.samples[idx]
        label = self.labels[idx]
        duration = self.durations[idx]

        # Construct full path
        if label == 1:  # English
            audio_path = DATA_DIR / "cv26-en" / "clips" / audio_filename
        else:  # Danish
            audio_path = DATA_DIR / "cv26-da" / audio_filename

        # Load and preprocess
        waveform = load_and_preprocess(audio_path, target_sr=16000)

        # Extract Log Mel-spectrogram
        log_mel_spec = extract_log_mel_spectrogram(
            waveform=waveform, sample_rate=16000, config=self.mel_config
        )

        # Convert to tensor and add channel dimension
        spectrogram = torch.from_numpy(log_mel_spec).float().unsqueeze(0)

        return spectrogram, label, duration


def load_checkpoint(path: Path) -> tuple[dict, dict]:
    """Load model checkpoint.

    Args:
        path: Path to the checkpoint file.

    Returns:
        Tuple of (model_state_dict, config_dict).
    """
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)

    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint.get("config", {})
    else:
        return checkpoint, {}


def collate_fn(
    batch: list, pad_value: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for variable-length spectrograms.

    Args:
        batch: List of (spectrogram, label, duration) tuples.
        pad_value: Value to use for padding.

    Returns:
        Tuple of (padded_spectrograms, labels, durations).
    """
    spectrograms = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    durations = torch.tensor([item[2] for item in batch], dtype=torch.float)

    # Pad spectrograms to max time length
    max_time = max(s.shape[2] for s in spectrograms)
    padded = []
    for spec in spectrograms:
        pad_len = max_time - spec.shape[2]
        if pad_len > 0:
            spec = torch.nn.functional.pad(spec, (0, pad_len), value=pad_value)
        padded.append(spec)

    return torch.stack(padded), labels, durations


def create_model(model_size: str) -> torch.nn.Module:
    """Create model based on size variant.

    Args:
        model_size: One of 'tiny', 'small', 'medium'.

    Returns:
        Instantiated model.
    """
    if model_size == "tiny":
        return create_tiny_cnn(num_languages=2)
    elif model_size == "medium":
        return create_medium_cnn(num_languages=2)
    else:
        return create_small_cnn(num_languages=2)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, dataloader: DataLoader, device: torch.device
) -> dict:
    """Evaluate the model.

    Args:
        model: The model to evaluate.
        dataloader: DataLoader with test samples.
        device: Device to run evaluation on.

    Returns:
        Dictionary with evaluation metrics.
    """
    model.eval()

    all_preds: list[int] = []
    all_labels: list[int] = []
    all_durations: list[float] = []

    for spectrogram, label, duration in tqdm(dataloader, desc="Evaluating"):
        spectrogram = spectrogram.to(device)
        label = label.to(device)

        logits = model(spectrogram)
        pred = logits.argmax(dim=1)

        all_preds.extend(pred.cpu().tolist())
        all_labels.extend(label.cpu().tolist())
        all_durations.extend(duration.tolist())

    # Overall accuracy
    correct = sum(p == label for p, label in zip(all_preds, all_labels))
    total = len(all_labels)
    overall_accuracy = correct / total

    # Per-language accuracy
    da_correct = da_total = en_correct = en_total = 0
    for pred, label in zip(all_preds, all_labels):
        if label == 0:
            da_total += 1
            if pred == label:
                da_correct += 1
        else:
            en_total += 1
            if pred == label:
                en_correct += 1

    da_accuracy = da_correct / da_total if da_total > 0 else 0.0
    en_accuracy = en_correct / en_total if en_total > 0 else 0.0

    # Accuracy by duration
    duration_groups: dict[str, list[tuple[int, int]]] = {
        "0-2s": [],
        "2-4s": [],
        "4-6s": [],
        "6+s": [],
    }
    for pred, label, dur in zip(all_preds, all_labels, all_durations):
        if dur < 2.0:
            duration_groups["0-2s"].append((pred, label))
        elif dur < 4.0:
            duration_groups["2-4s"].append((pred, label))
        elif dur < 6.0:
            duration_groups["4-6s"].append((pred, label))
        else:
            duration_groups["6+s"].append((pred, label))

    duration_accuracy = {}
    for group, pairs in duration_groups.items():
        if pairs:
            group_correct = sum(p == label for p, label in pairs)
            duration_accuracy[group] = group_correct / len(pairs)
        else:
            duration_accuracy[group] = 0.0

    # Confusion matrix
    confusion = [[0, 0], [0, 0]]
    for pred, label in zip(all_preds, all_labels):
        confusion[label][pred] += 1

    return {
        "overall_accuracy": overall_accuracy,
        "per_language_accuracy": {"da": da_accuracy, "en": en_accuracy},
        "accuracy_by_duration": duration_accuracy,
        "confusion_matrix": confusion,
        "total_samples": total,
    }


def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Phase 4b models")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=DATA_DIR / "test.csv",
        help="Path to test CSV file",
    )
    args = parser.parse_args()

    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    state_dict, config = load_checkpoint(Path(args.checkpoint))
    model_size = config.get("model_size", "small")

    # Create model
    model = create_model(model_size)
    model.load_state_dict(state_dict)
    model.to(DEVICE)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count:,}")
    logger.info(f"Model size (FP32): {param_count * 4 / 1024:.1f} KB")
    logger.info(f"Device: {DEVICE}")
    logger.info("")

    # Load dataset
    logger.info(f"Loading test set from {args.test_csv}")
    dataset = Phase4BDataset(args.test_csv)
    logger.info(f"Test samples: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate_fn(batch, pad_value=0.0),
    )

    # Evaluate
    logger.info("")
    metrics = evaluate(model, dataloader, DEVICE)

    # Print results
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Overall accuracy: {metrics['overall_accuracy'] * 100:.2f}%")
    logger.info(
        f"Danish accuracy:  {metrics['per_language_accuracy']['da'] * 100:.2f}%"
    )
    logger.info(
        f"English accuracy: {metrics['per_language_accuracy']['en'] * 100:.2f}%"
    )
    logger.info("")
    logger.info("Accuracy by duration:")
    for duration, acc in metrics["accuracy_by_duration"].items():
        logger.info(f"  {duration}: {acc * 100:.2f}%")
    logger.info("")
    logger.info(f"Confusion matrix: {metrics['confusion_matrix']}")
    logger.info("=" * 60)

    # Save results
    output_dir = Path(args.checkpoint).parent
    output_path = output_dir / "evaluation_results.json"

    results = {
        "checkpoint": str(args.checkpoint),
        "model_params": param_count,
        "model_size_kb": param_count * 4 / 1024,
        "test_samples": metrics["total_samples"],
        **metrics,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
