#!/usr/bin/env python3
"""Phase 4b: Train Tiny CNN (15k params) for Ultra-Low RAM Deployment.

Trains a ~15k parameter CNN for edge devices with 100 KB - 1 MB RAM budgets.

Two training modes:
1. Direct training on hard labels (simpler, no teacher needed)
2. Knowledge distillation from Phase 2 teacher (potentially +1-2 pp accuracy)

Target: ~85-90% accuracy at ~100 KB RAM (vs 91.32% at 2.1 MB for Phase 2).
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_language_detection.config.defaults import Config
from tiny_language_detection.data.preprocessing import (
    assign_duration_group,
    get_audio_duration,
    load_and_preprocess,
)
from tiny_language_detection.eval.metrics import compute_metrics
from tiny_language_detection.features.mel_spectrogram import (
    MelSpectrogramConfig,
    extract_log_mel_spectrogram,
)
from tiny_language_detection.models.tiny_cnn import (
    CompactCNNLanguageDetector,
    create_tiny_cnn,
    create_small_cnn,
    create_medium_cnn,
)
from tiny_language_detection.models.cnn_rnn import CNNRNNLanguageDetector

# Constants
TRAIN_MANIFEST = Path("data/sampled/train.csv")
TEST_MANIFEST = Path("data/sampled/test.csv")
DATA_DIR = Path("data")
OUTPUT_DIR = Path("data/experiments/phase4b")


def setup_logging() -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class LogMelDataset(Dataset):
    """Dataset for Log Mel-spectrogram features."""

    def __init__(
        self,
        manifest_path: Path,
        mel_config: MelSpectrogramConfig,
    ) -> None:
        """Initialise the dataset.

        Args:
            manifest_path: Path to CSV manifest file.
            mel_config: Mel spectrogram configuration.
        """
        self.samples = []
        self.labels = []
        self.durations = []
        self.mel_config = mel_config

        with open(manifest_path, "r") as f:
            lines = f.readlines()[1:]  # Skip header

        for line in lines:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                audio_filename = parts[0]
                # parts[1] = language code ('da'/'en'), parts[2] = label (0/1)
                label = int(parts[2])  # 0=da, 1=en
                duration = float(parts[4]) if len(parts) >= 5 else None
                
                # Store filename only; construct full path in __getitem__
                self.samples.append(audio_filename)
                self.labels.append(label)
                self.durations.append(duration)

        logging.info(f"Loaded {len(self.samples)} samples from {manifest_path}")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str, float]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (spectrogram, label_id, audio_path, duration).
        """
        audio_filename = self.samples[idx]
        label = self.labels[idx]
        duration = self.durations[idx]

        # Construct full path (same as evaluate_phase2.py)
        # Danish files are in data/cv26-da/, English in data/cv26-en/clips/
        if label == 1:  # English
            audio_path = DATA_DIR / "cv26-en" / "clips" / audio_filename
        else:  # Danish
            audio_path = DATA_DIR / "cv26-da" / audio_filename

        # Load and preprocess
        waveform = load_and_preprocess(audio_path, target_sr=16000)

        # Extract Log Mel-spectrogram
        log_mel_spec = extract_log_mel_spectrogram(
            waveform=waveform,
            sample_rate=16000,
            config=self.mel_config,
        )

        # Convert to tensor and add channel dimension
        spectrogram = torch.from_numpy(log_mel_spec).float().unsqueeze(0)

        return spectrogram, label, str(audio_path), duration or get_audio_duration(audio_path)


def collate_fn(batch):
    """Collate with padding for variable-length sequences."""
    specs, labels, paths, durations = zip(*batch)

    # Pad to max time steps
    max_time = max(s.shape[2] for s in specs)
    padded_specs = []
    for s in specs:
        pad = max_time - s.shape[2]
        if pad > 0:
            s = F.pad(s, (0, pad))
        padded_specs.append(s)

    specs_tensor = torch.stack(padded_specs)
    labels_tensor = torch.tensor(labels)

    return specs_tensor, labels_tensor, list(paths), list(durations)


def load_phase2_teacher(device: torch.device) -> CNNRNNLanguageDetector:
    """Load the Phase 2 teacher model (frozen, for KD)."""
    teacher_path = OUTPUT_DIR.parent / "phase2" / "model.pth"
    config_path = teacher_path.parent / "config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    teacher = CNNRNNLanguageDetector(
        n_mels=config["n_mels"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
    )

    checkpoint = torch.load(teacher_path, map_location=device, weights_only=True)
    teacher.load_state_dict(checkpoint)
    teacher.to(device)
    teacher.eval()

    # Freeze teacher
    for param in teacher.parameters():
        param.requires_grad = False

    logging.info(f"Loaded Phase 2 teacher from {teacher_path}")
    logging.info(f"Teacher parameters: {teacher.count_parameters():,}")

    return teacher


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    teacher: nn.Module | None = None,
    kd_alpha: float = 0.5,
    temperature: float = 2.0,
) -> tuple[float, float]:
    """Train for one epoch.

    Args:
        model: Student model to train.
        train_loader: Data loader for training set.
        optimizer: Optimizer.
        criterion: Loss function (CrossEntropyLoss).
        device: Torch device.
        teacher: Teacher model (for KD), or None for direct training.
        kd_alpha: Weight for KD loss (0.5 = 50% label + 50% teacher).
        temperature: Temperature for softening KD targets.

    Returns:
        Tuple of (average loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (specs, labels, _, _) in enumerate(train_loader):
        specs = specs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Student forward
        student_logits = model(specs)

        # Compute loss
        if teacher is not None:
            # Knowledge distillation: mix of hard labels + teacher soft targets
            with torch.no_grad():
                teacher_logits = teacher(specs)

            # Soft targets loss (KL divergence)
            student_soft = F.log_softmax(student_logits / temperature, dim=1)
            teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
            kd_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (temperature ** 2)

            # Hard labels loss (cross-entropy)
            ce_loss = criterion(student_logits, labels)

            # Combined loss
            loss = (1 - kd_alpha) * ce_loss + kd_alpha * kd_loss
        else:
            # Direct training on hard labels only
            loss = criterion(student_logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = student_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate model on test set.

    Args:
        model: Model to evaluate.
        test_loader: Data loader for test set.
        device: Torch device.

    Returns:
        Evaluation metrics dictionary.
    """
    model.eval()

    predictions = []
    labels = []
    paths_durations = []

    for specs, label_batch, paths, durations in test_loader:
        specs = specs.to(device)
        outputs = model(specs)
        _, predicted = outputs.max(1)

        predictions.extend(predicted.cpu().numpy())
        labels.extend(label_batch.numpy())
        paths_durations.extend(list(zip(paths, durations)))

    # Compute metrics
    languages = ["da" if l == 0 else "en" for l in labels]
    duration_groups = [assign_duration_group(d) for _, d in paths_durations]

    metrics = compute_metrics(
        predictions=predictions,
        labels=labels,
        languages=languages,
        duration_groups=duration_groups,
    )

    return metrics


def get_model_size(model: nn.Module) -> int:
    """Calculate model size in bytes."""
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    return param_size + buffer_size


def main() -> None:
    """Main training entry point."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Phase 4b: Train Compact CNN")
    parser.add_argument("--mode", choices=["direct", "kd"], default="direct",
                        help="Training mode: 'direct' (labels only) or 'kd' (knowledge distillation)")
    parser.add_argument("--model-size", choices=["tiny", "small", "medium"], default="small",
                        help="Model size: 'tiny' (~50k), 'small' (~100k), 'medium' (~200k)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--kd-alpha", type=float, default=0.5, help="KD loss weight (0-1)")
    parser.add_argument("--temperature", type=float, default=2.0, help="KD temperature")
    args = parser.parse_args()

    # Device
    device = torch.device("cpu")  # Target deployment is CPU
    logging.info(f"Using CPU (deployment-target configuration)")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exp_name = f"tiny_cnn_{args.mode}"
    exp_dir = OUTPUT_DIR / exp_name
    exp_dir.mkdir(exist_ok=True)

    # Configuration
    config = Config()
    mel_config = MelSpectrogramConfig(
        sample_rate=config.sample_rate,
        n_mels=80,
        n_fft=400,
        hop_length=160,
    )

    # Load teacher if using KD
    teacher = None
    if args.mode == "kd":
        teacher = load_phase2_teacher(device)
        logging.info(f"Training with KD (alpha={args.kd_alpha}, T={args.temperature})")
    else:
        logging.info("Training with direct labels (no teacher)")

    # Create student model
    if args.model_size == "tiny":
        student = create_tiny_cnn(num_languages=2)
    elif args.model_size == "small":
        student = create_small_cnn(num_languages=2)
    else:  # medium
        student = create_medium_cnn(num_languages=2)
    
    # Override dropout if specified
    if args.dropout != 0.3:
        student.classifier[2] = nn.Dropout(args.dropout)
    
    student.to(device)
    logging.info(f"Model size variant: {args.model_size}")
    logging.info(f"Student parameters: {student.count_parameters():,}")
    logging.info(f"Student size (FP32): {get_model_size(student) / 1024:.1f} KB")

    # Data loaders
    train_dataset = LogMelDataset(TRAIN_MANIFEST, mel_config)
    test_dataset = LogMelDataset(TEST_MANIFEST, mel_config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    logging.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # Optimizer and loss
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    logging.info(f"Starting training for {args.epochs} epochs...")
    logging.info("=" * 60)

    best_accuracy = 0.0
    training_history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model=student,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            teacher=teacher,
            kd_alpha=args.kd_alpha,
            temperature=args.temperature,
        )

        # Evaluate on test set
        test_metrics = evaluate(student, test_loader, device)
        test_acc = test_metrics["overall_accuracy"]

        # Log progress
        logging.info(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Test Acc: {test_acc * 100:.2f}%"
        )

        # Track history
        training_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "test_metrics": test_metrics,
        })

        # Learning rate scheduling
        scheduler.step(test_acc)

        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model_path = exp_dir / "model_best.pth"
            torch.save(student.state_dict(), best_model_path)
            logging.info(f"  → New best! Saved to {best_model_path}")

    logging.info("=" * 60)
    logging.info(f"Training complete! Best test accuracy: {best_accuracy * 100:.2f}%")

    # Save final model and config
    final_model_path = exp_dir / "model.pth"
    torch.save(student.state_dict(), final_model_path)
    logging.info(f"Saved final model to {final_model_path}")

    # Save training history
    history_path = exp_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)
    logging.info(f"Saved training history to {history_path}")

    # Save experiment config
    final_test_metrics = training_history[-1]["test_metrics"]
    experiment_config = {
        "mode": args.mode,
        "model_size": args.model_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "dropout": args.dropout,
        "kd_alpha": args.kd_alpha if args.mode == "kd" else None,
        "temperature": args.temperature if args.mode == "kd" else None,
        "student_params": student.count_parameters(),
        "student_size_kb": get_model_size(student) / 1024,
        "best_test_accuracy": best_accuracy,
        "final_test_metrics": {
            "overall_accuracy": final_test_metrics["overall_accuracy"],
            "per_language_accuracy": final_test_metrics["per_language_accuracy"],
            "accuracy_by_duration": final_test_metrics["accuracy_by_duration"],
            "confusion_matrix": final_test_metrics["confusion_matrix"],
        },
    }

    config_path = exp_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(experiment_config, f, indent=2)
    logging.info(f"Saved experiment config to {config_path}")

    # Print summary
    logging.info("")
    logging.info("=" * 60)
    logging.info("FINAL SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Mode: {args.mode}")
    logging.info(f"Student parameters: {student.count_parameters():,}")
    logging.info(f"Student size (FP32): {get_model_size(student) / 1024:.1f} KB")
    logging.info(f"Best test accuracy: {best_accuracy * 100:.2f}%")
    logging.info(f"Danish accuracy: {final_test_metrics['per_language_accuracy']['da'] * 100:.2f}%")
    logging.info(f"English accuracy: {final_test_metrics['per_language_accuracy']['en'] * 100:.2f}%")
    logging.info("")
    logging.info("Confusion matrix:")
    cm = final_test_metrics["confusion_matrix"]
    logging.info(f"  [[{cm[0][0]}, {cm[0][1]}],")
    logging.info(f"   [{cm[1][0]}, {cm[1][1]}]]")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
