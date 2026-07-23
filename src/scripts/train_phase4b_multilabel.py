#!/usr/bin/env python3
"""Train Phase 4b Compact CNN with Multi-Label Output + SpecAugment."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tiny_language_detection.features.mel_spectrogram import MelSpectrogramConfig
from tiny_language_detection.features.spec_augment import SpecAugment
from tiny_language_detection.data.preprocessing import load_and_preprocess
from tiny_language_detection.models.tiny_cnn import create_small_cnn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultilabelDataset:
    """Dataset for multi-label training."""
    
    def __init__(
        self,
        manifest_path: Path,
        mel_config: MelSpectrogramConfig,
        time_mask_param: int = 5,
        freq_mask_param: int = 4,
        use_augment: bool = True,
    ) -> None:
        self.samples = []
        self.labels = []
        self.mel_config = mel_config
        self.use_augment = use_augment
        
        self.augment = SpecAugment(
            time_mask_param=time_mask_param,
            freq_mask_param=freq_mask_param,
            time_masks=1,
            freq_masks=1,
        )

        with open(manifest_path, "r") as f:
            lines = f.readlines()[1:]

        for line in lines:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                audio_filename = parts[0]
                language = parts[1]
                label = [1.0, 0.0] if language == "da" else [0.0, 1.0]
                self.samples.append((audio_filename, language))
                self.labels.append(label)

        logger.info(f"Loaded {len(self.samples)} samples from {manifest_path}")
        if use_augment:
            logger.info(f"SpecAugment: time_mask={time_mask_param}, freq_mask={freq_mask_param}")

    def __len__(self) -> int:
        return len(self.samples)

    def _get_audio_path(self, filename: str, language: str) -> Path:
        if language == "da":
            return DATA_DIR / "cv26-da" / filename
        else:
            return DATA_DIR / "cv26-en" / "clips" / filename

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        filename, language = self.samples[idx]
        audio_path = self._get_audio_path(filename, language)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        waveform = load_and_preprocess(str(audio_path), target_sr=16000)
        from tiny_language_detection.features.mel_spectrogram import extract_log_mel_spectrogram
        log_mel = extract_log_mel_spectrogram(waveform, 16000, self.mel_config)
        spec = torch.from_numpy(log_mel).float()
        
        if self.use_augment:
            spec = self.augment(spec.unsqueeze(0)).squeeze(0)
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return spec, label, language


def collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    specs, labels, _ = zip(*batch)
    max_time = max(s.shape[1] for s in specs)
    padded_specs = []
    for spec in specs:
        pad = torch.zeros(spec.shape[0], max_time - spec.shape[1])
        padded = torch.cat([spec, pad], dim=1)
        padded_specs.append(padded)
    return torch.stack(padded_specs).unsqueeze(1), torch.stack(labels)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    max_grad_norm: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    clips = 0

    for specs, labels in tqdm(loader, desc="Training"):
        specs = specs.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(specs)
        loss = criterion(logits, labels)
        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if grad_norm > max_grad_norm:
            clips += 1
        
        optimizer.step()
        
        total_loss += loss.item() * specs.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).all(dim=1).sum().item()
        total += specs.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "grad_clips_pct": (clips / len(loader)) * 100,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    da_correct = da_total = 0
    en_correct = en_total = 0

    for specs, labels in tqdm(loader, desc="Evaluating"):
        specs = specs.to(DEVICE)
        labels = labels.to(DEVICE)
        logits = model(specs)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        total_loss += loss.item() * specs.size(0)
        correct += (preds == labels).all(dim=1).sum().item()
        total += specs.size(0)
        
        for i in range(specs.size(0)):
            if labels[i, 0] > 0.5:
                da_total += 1
                if preds[i, 0] > 0.5:
                    da_correct += 1
            else:
                en_total += 1
                if preds[i, 1] > 0.5:
                    en_correct += 1

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "da_accuracy": da_correct / da_total if da_total > 0 else 0,
        "en_accuracy": en_correct / en_total if en_total > 0 else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase 4b Multi-Label CNN")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--time-mask", type=int, default=3)
    parser.add_argument("--freq-mask", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("data/experiments/phase4b/tiny_cnn_multilabel_v3"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Training Configuration (Generalization Focus):")
    logger.info(f"  Learning rate: {args.lr} (+ weight_decay={args.weight_decay})")
    logger.info(f"  Max grad norm: {args.max_grad_norm}")
    logger.info(f"  SpecAugment: time={args.time_mask}, freq={args.freq_mask} (conservative)")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info("=" * 60)

    mel_config = MelSpectrogramConfig()
    train_dataset = MultilabelDataset(
        DATA_DIR / "sampled/train.csv", mel_config,
        time_mask_param=args.time_mask,
        freq_mask_param=args.freq_mask,
        use_augment=True,
    )
    test_dataset = MultilabelDataset(
        DATA_DIR / "sampled/test.csv", mel_config,
        time_mask_param=0,
        freq_mask_param=0,
        use_augment=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = create_small_cnn(num_languages=2)
    model.to(DEVICE)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_accuracy = 0.0
    best_epoch = 0
    history = []
    stable_epochs = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, args.max_grad_norm)
        eval_metrics = evaluate(model, test_loader, criterion)

        result = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "grad_clips_pct": train_metrics["grad_clips_pct"],
            "test_loss": eval_metrics["loss"],
            "test_accuracy": eval_metrics["accuracy"],
            "da_accuracy": eval_metrics["da_accuracy"],
            "en_accuracy": eval_metrics["en_accuracy"],
            "lr": optimizer.param_groups[0]['lr'],
        }
        history.append(result)

        # Track stable epochs (both languages > 85%)
        is_stable = eval_metrics["da_accuracy"] > 0.85 and eval_metrics["en_accuracy"] > 0.85
        if is_stable:
            stable_epochs.append(epoch)

        logger.info(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"LR: {optimizer.param_groups[0]['lr']:.1e} | "
            f"Clips: {train_metrics['grad_clips_pct']:.1f}% | "
            f"Train: {train_metrics['accuracy']:.2%} | "
            f"Test: {eval_metrics['accuracy']:.2%} (DA: {eval_metrics['da_accuracy']:.2%}, EN: {eval_metrics['en_accuracy']:.2%})"
            f"{' ✓' if is_stable else ''}"
        )

        scheduler.step()

        if eval_metrics["accuracy"] > best_accuracy:
            best_accuracy = eval_metrics["accuracy"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "accuracy": best_accuracy,
                "da_accuracy": eval_metrics["da_accuracy"],
                "en_accuracy": eval_metrics["en_accuracy"],
            }, args.output_dir / "checkpoint_best.pth")
            torch.save(model.state_dict(), args.output_dir / "model_best.pth")
            logger.info(f"  → Best! Epoch {epoch}, Acc: {best_accuracy:.2%}")

    # Save ensemble of stable epochs
    if len(stable_epochs) >= 3:
        logger.info(f"\nFound {len(stable_epochs)} stable epochs: {stable_epochs}")
        # For now, just note them - ensemble would require storing multiple checkpoints

    with open(args.output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info("")
    logger.info(f"Complete! Best: {best_accuracy:.2%} (epoch {best_epoch})")
    logger.info(f"Stable epochs: {stable_epochs}")


if __name__ == "__main__":
    main()
