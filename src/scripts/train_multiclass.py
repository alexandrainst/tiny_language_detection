#!/usr/bin/env python3
"""Train Compact CNN for multi-class language detection (2–23 languages).

Supports:
- Binary (Danish vs English) from Common Voice
- Multi-class (23 languages) from YODAS-Granary
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tiny_language_detection.data.preprocessing import load_and_preprocess
from tiny_language_detection.features.mel_spectrogram import (
    MelSpectrogramConfig,
    extract_log_mel_spectrogram,
)
from tiny_language_detection.features.spec_augment import SpecAugment
from tiny_language_detection.models.tiny_cnn import CompactCNNLanguageDetector

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ISO 639-1 language codes for 23 languages
LANG_CODES = [
    "bg",
    "hr",
    "cs",
    "da",
    "nl",
    "en",
    "et",
    "fi",
    "fr",
    "de",
    "el",
    "hu",
    "it",
    "lv",
    "lt",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "es",
    "sv",
    "uk",
]
LANG_TO_ID = {code: i for i, code in enumerate(LANG_CODES)}


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
    ) -> None:
        self.samples = []
        self.labels = []
        self.languages = []
        self.mel_config = mel_config
        self.use_augment = use_augment
        self.use_hf = use_hf

        self.augment = SpecAugment(
            time_mask_param=time_mask_param,
            freq_mask_param=freq_mask_param,
            time_masks=1,
            freq_masks=1,
        )

        if use_hf:
            # Load from HuggingFace datasets
            logger.info(f"Loading HF dataset: {source}, split={hf_split}")
            ds = load_dataset(source, split=hf_split, streaming=False)

            for sample in ds:
                lang = sample["lang"]
                if lang not in LANG_TO_ID:
                    continue  # Skip unknown languages
                self.samples.append(sample["audio"])
                self.labels.append(LANG_TO_ID[lang])
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
                    if language not in LANG_TO_ID:
                        continue
                    self.samples.append((audio_filename, language))
                    self.labels.append(LANG_TO_ID[language])
                    self.languages.append(language)

            logger.info(f"Loaded {len(self.samples)} samples from manifest")

        if use_augment:
            logger.info(
                f"SpecAugment: time_mask={time_mask_param}, freq_mask={freq_mask_param}"
            )
        logger.info(
            f"Languages: {len(set(self.languages))} ({sorted(set(self.languages))})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _get_audio_path(self, filename: str, language: str) -> Path:
        """Get audio file path for manifest-based loading."""
        # Support both CV structure and generic data/
        if language == "da":
            return Path("data") / "cv26-da" / filename
        elif language == "en":
            return Path("data") / "cv26-en" / "clips" / filename
        else:
            return Path("data") / "audio" / language / filename

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        if self.use_hf:
            audio_dict = self.samples[idx]
            waveform = torch.from_numpy(audio_dict["array"]).float()
            if len(waveform.shape) == 2:
                waveform = waveform.squeeze(0)
            sr = audio_dict.get("sampling_rate", 16000)
        else:
            filename, language = self.samples[idx]
            audio_path = self._get_audio_path(filename, language)

            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            waveform = load_and_preprocess(str(audio_path), target_sr=16000)
            sr = 16000

        # Extract log-Mel spectrogram
        log_mel = extract_log_mel_spectrogram(waveform, sr, self.mel_config)
        spec = torch.from_numpy(log_mel).float()

        if self.use_augment:
            spec = self.augment(spec.unsqueeze(0)).squeeze(0)

        label = self.labels[idx]
        return spec, label, self.languages[idx]


def collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function for variable-length spectrograms."""
    specs, labels, langs = zip(*batch)
    max_time = max(s.shape[1] for s in specs)
    padded_specs = []
    for spec in specs:
        pad = torch.zeros(spec.shape[0], max_time - spec.shape[1])
        padded = torch.cat([spec, pad], dim=1)
        padded_specs.append(padded)
    return torch.stack(padded_specs).unsqueeze(1), torch.tensor(
        labels, dtype=torch.long
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    max_grad_norm: float,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    grad_clips = 0
    total_steps = 0

    for specs, labels in tqdm(loader, desc="Training"):
        specs = specs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(specs)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping
        total_steps += 1
        if max_grad_norm > 0:
            grad_clipped = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            )
            if grad_clipped.item() >= max_grad_norm:
                grad_clips += 1

        optimizer.step()

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * specs.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "grad_clips_pct": (grad_clips / total_steps) * 100 if total_steps > 0 else 0,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    lang_to_id: dict[str, int] = None,
) -> dict[str, Any]:
    """Evaluate on test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # Per-language accuracy
    lang_correct = {}
    lang_total = {}

    id_to_lang = {v: k for k, v in (lang_to_id or LANG_TO_ID).items()}

    for specs, labels in tqdm(loader, desc="Evaluating"):
        specs = specs.to(DEVICE)
        labels = labels.to(DEVICE)
        logits = model(specs)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * specs.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Per-language stats
        for i in range(specs.size(0)):
            lang_id = labels[i].item()
            lang = id_to_lang.get(lang_id, str(lang_id))
            lang_total[lang] = lang_total.get(lang, 0) + 1
            if preds[i].item() == lang_id:
                lang_correct[lang] = lang_correct.get(lang, 0) + 1

    # Calculate per-language accuracy
    lang_accuracy = {
        lang: lang_correct.get(lang, 0) / count for lang, count in lang_total.items()
    }

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "lang_accuracy": lang_accuracy,
        "lang_total": lang_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Multi-Class Language Detection CNN"
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--time-mask", type=int, default=3)
    parser.add_argument("--freq-mask", type=int, default=2)
    parser.add_argument(
        "--num-languages",
        type=int,
        default=23,
        help="Number of output classes (2 or 23)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="saattrupdan/yodas-granary-language-detection",
        help="HF dataset name or path to manifest CSV",
    )
    parser.add_argument(
        "--use-hf", action="store_true", help="Load from HuggingFace datasets"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/experiments/multiclass")
    )
    args = parser.parse_args()

    num_classes = args.num_languages
    assert num_classes in [2, 23], f"num_languages must be 2 or 23, got {num_classes}"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Training Configuration:")
    logger.info(f"  Dataset: {args.dataset} (HF: {args.use_hf})")
    logger.info(f"  Num languages: {num_classes}")
    logger.info(f"  Learning rate: {args.lr} (+ weight_decay={args.weight_decay})")
    logger.info(f"  Max grad norm: {args.max_grad_norm}")
    logger.info(f"  SpecAugment: time={args.time_mask}, freq={args.freq_mask}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info("=" * 60)

    mel_config = MelSpectrogramConfig()
    train_dataset = MulticlassDataset(
        args.dataset,
        mel_config,
        time_mask_param=args.time_mask,
        freq_mask_param=args.freq_mask,
        use_augment=True,
        use_hf=args.use_hf,
        hf_split="train",
    )
    test_dataset = MulticlassDataset(
        args.dataset,
        mel_config,
        time_mask_param=0,
        freq_mask_param=0,
        use_augment=False,
        use_hf=args.use_hf,
        hf_split="test",
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Create model with configurable num_languages
    model = CompactCNNLanguageDetector(
        n_mels=mel_config.n_mels,
        channels=[32, 64, 128],
        hidden_size=128,
        num_languages=num_classes,
        dropout=0.3,
    )
    model.to(DEVICE)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_accuracy = 0.0
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, args.max_grad_norm
        )
        eval_metrics = evaluate(
            model,
            test_loader,
            criterion,
            LANG_TO_ID if num_classes == 23 else {"da": 0, "en": 1},
        )

        result = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "grad_clips_pct": train_metrics["grad_clips_pct"],
            "test_loss": eval_metrics["loss"],
            "test_accuracy": eval_metrics["accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
        }

        # Add per-language accuracy for binary case
        if num_classes == 2:
            lang_acc = eval_metrics.get("lang_accuracy", {})
            result["da_accuracy"] = lang_acc.get("da", 0)
            result["en_accuracy"] = lang_acc.get("en", 0)

        history.append(result)

        # Build log message
        log_msg = (
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"LR: {optimizer.param_groups[0]['lr']:.1e} | "
            f"Clips: {train_metrics['grad_clips_pct']:.1f}% | "
            f"Train: {train_metrics['accuracy']:.2%} | "
            f"Test: {eval_metrics['accuracy']:.2%}"
        )

        if num_classes == 2:
            lang_acc = eval_metrics.get("lang_accuracy", {})
            log_msg += (
                f" (DA: {lang_acc.get('da', 0):.2%}, EN: {lang_acc.get('en', 0):.2%})"
            )
        elif num_classes == 23:
            # Show top 5 languages by accuracy and bottom 5
            lang_acc = eval_metrics.get("lang_accuracy", {})
            sorted_langs = sorted(lang_acc.items(), key=lambda x: x[1], reverse=True)
            top5 = sorted_langs[:5]
            bottom5 = sorted_langs[-5:]
            top_str = ", ".join([f"{lang}:{acc:.0%}" for lang, acc in top5])
            bottom_str = ", ".join([f"{lang}:{acc:.0%}" for lang, acc in bottom5])
            log_msg += f" | Top: {top_str}, Bottom: {bottom_str}"

        logger.info(log_msg)

        scheduler.step()

        if eval_metrics["accuracy"] > best_accuracy:
            best_accuracy = eval_metrics["accuracy"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "accuracy": best_accuracy,
                    "num_languages": num_classes,
                    "lang_accuracy": eval_metrics.get("lang_accuracy", {}),
                },
                args.output_dir / "checkpoint_best.pth",
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_languages": num_classes,
                    "mel_config": mel_config,
                },
                args.output_dir / "model_best.pth",
            )
            logger.info(f"  → Best! Epoch {epoch}, Acc: {best_accuracy:.2%}")

    # Save full per-language breakdown for 23-class
    if num_classes == 23:
        final_eval = evaluate(model, test_loader, criterion, LANG_TO_ID)
        with open(args.output_dir / "per_language_accuracy.json", "w") as f:
            json.dump(
                {
                    "best_epoch": best_epoch,
                    "best_accuracy": best_accuracy,
                    "per_language": final_eval["lang_accuracy"],
                    "per_language_counts": final_eval["lang_total"],
                },
                f,
                indent=2,
            )
        logger.info(
            f"Per-language accuracy saved to {args.output_dir / 'per_language_accuracy.json'}"  # noqa: E501
        )

    with open(args.output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info("")
    logger.info(f"Complete! Best: {best_accuracy:.2%} (epoch {best_epoch})")
    logger.info(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
