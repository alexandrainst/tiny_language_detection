"""Shared training utilities for multi-class language detection."""

import json
import logging
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tiny_language_detection.features.mel_spectrogram import MelSpectrogramConfig

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_class_weights(labels: list[int]) -> torch.Tensor:
    """Compute class weights for imbalanced datasets.

    Uses inverse frequency weighting: weight_c = total / (num_classes × count_c)
    """
    label_counts = Counter(labels)
    num_classes = len(label_counts)
    total_samples = sum(label_counts.values())

    weights = []
    for class_id in range(num_classes):
        count = label_counts.get(class_id, 0)
        if count > 0:
            weight = total_samples / (num_classes * count)
        else:
            weight = 1.0
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    max_grad_norm: float,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    grad_clips = 0
    total_steps = 0

    for specs, labels, _ in tqdm(train_loader, desc="Training", leave=False):
        specs = specs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        if max_grad_norm > 0:
            grad_clipped = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            )
            if grad_clipped.item() >= max_grad_norm:
                grad_clips += 1

        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        total_steps += 1

    return {
        "loss": total_loss / total_steps,
        "accuracy": correct / total,
        "grad_clips_pct": (grad_clips / total_steps) * 100 if total_steps > 0 else 0,
    }


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    languages: list[str],
) -> dict:
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    lang_correct = Counter()
    lang_total = Counter()

    with torch.no_grad():
        for specs, labels, lang_names in tqdm(
            test_loader, desc="Evaluating", leave=False
        ):
            specs = specs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(specs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Per-language statistics
            for i, lang in enumerate(lang_names):
                lang_total[lang] += 1
                if predicted[i].item() == labels[i].item():
                    lang_correct[lang] += 1

    # Per-language accuracy
    lang_accuracy = {
        lang: lang_correct[lang] / lang_total[lang]
        for lang in lang_total
        if lang_total[lang] > 0
    }

    return {
        "loss": total_loss / len(test_loader),
        "accuracy": correct / total,
        "lang_accuracy": lang_accuracy,
    }


def save_checkpoint(
    state: dict, output_dir: Path, filename: str = "checkpoint.pth.tar"
) -> None:
    """Save training checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_dir / filename)


def save_best_model(
    model: nn.Module, mel_config: MelSpectrogramConfig, output_dir: Path
) -> None:
    """Save best model with config."""
    from tiny_language_detection.models.cnn_rnn import CNNRNNLanguageDetector
    from tiny_language_detection.models.tiny_cnn import CompactCNNLanguageDetector

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), output_dir / "best_model.pth")

    # Save config
    if isinstance(model, CompactCNNLanguageDetector):
        config = {
            "model_type": "compact_cnn",
            "n_mels": model.n_mels,
            "channels": model.channels,
            "hidden_size": model.classifier[0].in_features,  # Approximate
            "num_languages": model.num_languages,
        }
    elif isinstance(model, CNNRNNLanguageDetector):
        config = {
            "model_type": "cnn_rnn",
            "n_mels": model.n_mels,
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
            "num_languages": model.num_languages,
        }
    else:
        config = {"model_type": "unknown"}

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Saved best model to {output_dir}")
