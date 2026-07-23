#!/usr/bin/env python3
"""Train Compact CNN for multi-class language detection (N classes).

Supports:
- Binary classification (e.g., Danish vs English)
- Multi-class classification (e.g., 23 languages from YODAS-Granary)
- Custom N-language datasets from HuggingFace or local manifests
"""

import argparse
import json
import logging
from collections import Counter
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
        data_dir: Path = None,
        time_masks: int = 1,
        freq_masks: int = 1,
    ) -> None:
        self.samples = []
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

        if use_augment:
            logger.info(
                f"SpecAugment: time_mask={time_mask_param} (×{time_masks}), "
                f"freq_mask={freq_mask_param} (×{freq_masks})"
            )
        logger.info(
            f"Languages: {len(set(self.languages))} ({sorted(set(self.languages))})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _get_audio_path(self, filename: str, language: str) -> Path:
        """Get audio file path for manifest-based loading.

        Expects data directory structure:
        {data_dir}/
          da/
            audio1.wav
          en/
            audio2.wav
        """
        return self.data_dir / language / filename

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

    id_to_lang = {v: k for k, v in lang_to_id.items()} if lang_to_id else {}

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
        "--time-masks", type=int, default=1, help="Number of time masks (default: 1)"
    )
    parser.add_argument(
        "--freq-masks",
        type=int,
        default=1,
        help="Number of frequency masks (default: 1)",
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=80,
        help="Number of Mel filter banks (default: 80)",
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=400,
        help="FFT window size (default: 400)",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=160,
        help="Hop length between frames (default: 160)",
    )
    parser.add_argument(
        "--f-min",
        type=float,
        default=0.0,
        help="Minimum frequency in Hz (default: 0.0)",
    )
    parser.add_argument(
        "--f-max",
        type=float,
        default=None,
        help="Maximum frequency in Hz (default: None = sr/2)",
    )
    parser.add_argument(
        "--num-languages",
        type=int,
        default=None,
        help="Number of output classes (auto-detected from dataset if not specified)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs=3,
        default=[32, 64, 128],
        metavar=("C1", "C2", "C3"),
        help="CNN channels per block (default: 32 64 128)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="Classifier hidden layer size (default: 128)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout probability (default: 0.3)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        required=True,
        help="HF dataset name (with --use-hf) or path to manifest CSV",
    )
    parser.add_argument(
        "--use-hf", action="store_true", help="Load from HuggingFace datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for checkpoints and logs (default: data/experiments/{dataset_name})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Base directory containing language subdirectories (default: data/)",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class weights (enabled by default for N > 2 classes)",
    )
    args = parser.parse_args()

    # Class weights enabled by default for multi-class (N > 2)
    use_class_weights = not args.no_class_weights

    # Create default output directory based on dataset name
    if args.output_dir is None:
        dataset_name = Path(args.dataset).name if args.use_hf else Path(args.dataset).stem
        args.output_dir = Path("data/experiments") / dataset_name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mel_config = MelSpectrogramConfig(
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        f_min=args.f_min,
        f_max=args.f_max,
    )
    train_dataset = MulticlassDataset(
        args.dataset,
        mel_config,
        time_mask_param=args.time_mask,
        freq_mask_param=args.freq_mask,
        use_augment=True,
        use_hf=args.use_hf,
        hf_split="train",
        data_dir=args.data_dir,
        time_masks=args.time_masks,
        freq_masks=args.freq_masks,
    )

    # Auto-detect num_classes from dataset if not specified
    if args.num_languages is None:
        num_classes = len(set(train_dataset.languages))
    else:
        num_classes = args.num_languages

    logger.info("=" * 60)
    logger.info("Training Configuration:")
    logger.info(f"  Dataset: {args.dataset} (HF: {args.use_hf})")
    logger.info(f"  Num languages: {num_classes}")
    logger.info(
        f"  Num languages: {num_classes} "
        f"({"auto-detected" if args.num_languages is None else "specified"})"
    )
    logger.info(f"  Class weights: {use_class_weights}")
    logger.info(f"  CNN channels: {args.channels}")
    logger.info(f"  Hidden size: {args.hidden_size}")
    logger.info(f"  Dropout: {args.dropout}")
    logger.info(f"  Mel spectrogram: n_mels={args.n_mels}, n_fft={args.n_fft}, hop={args.hop_length}")
    logger.info(f"  Learning rate: {args.lr} (+ weight_decay={args.weight_decay})")
    logger.info(f"  Max grad norm: {args.max_grad_norm}")
    logger.info(
        f"  SpecAugment: time={args.time_mask} (×{args.time_masks}), "
        f"freq={args.freq_mask} (×{args.freq_masks})"
    )
    logger.info(f"  Epochs: {args.epochs}")
    logger.info("=" * 60)

    test_dataset = MulticlassDataset(
        args.dataset,
        mel_config,
        time_mask_param=0,
        freq_mask_param=0,
        use_augment=False,
        use_hf=args.use_hf,
        hf_split="test",
        data_dir=args.data_dir,
        time_masks=0,
        freq_masks=0,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Create model with configurable architecture
    model = CompactCNNLanguageDetector(
        n_mels=mel_config.n_mels,
        channels=args.channels,
        hidden_size=args.hidden_size,
        num_languages=num_classes,
        dropout=args.dropout,
    )
    model.to(DEVICE)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Compute class weights for imbalanced datasets
    # Compute class weights for imbalanced datasets
    class_weights_tensor = None
    if use_class_weights and num_classes > 2:
        lang_counts = Counter(train_dataset.languages)
        total_samples = len(train_dataset)
        # Get unique languages in sorted order for consistent indexing
        unique_langs = sorted(set(train_dataset.languages))
        # Inverse frequency weighting: weight_c = total / (num_classes * count_c)
        class_weights_list = [
            total_samples / (num_classes * lang_counts.get(lang, 1))
            for lang in unique_langs
        ]
        class_weights_tensor = torch.tensor(class_weights_list, dtype=torch.float32).to(
            DEVICE
        )
        logger.info(
            f"Class weights: min={class_weights_tensor.min():.2f}, "
            f"max={class_weights_tensor.max():.2f}, "
            f"mean={class_weights_tensor.mean():.2f}"
        )
        # Log extreme weights
        weight_lang_pairs = list(zip(class_weights_list, unique_langs))
        top_weighted = sorted(weight_lang_pairs, key=lambda x: -x[0])[:3]
        logger.info(
            f"Highest weights: {', '.join([f'{lang}:{w:.1f}' for w, lang in top_weighted])}"
        )

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
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
        # Build lang_to_id mapping dynamically from dataset
        unique_langs = sorted(set(train_dataset.languages))
        lang_to_id = {lang: i for i, lang in enumerate(unique_langs)}

        eval_metrics = evaluate(
            model,
            test_loader,
            criterion,
            lang_to_id,
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

        history.append(result)

        # Build log message
        lang_acc = eval_metrics.get("lang_accuracy", {})

        if num_classes <= 5:
            # Show all languages for small N
            lang_details = ", ".join(
                [f"{lang}:{acc:.0%}" for lang, acc in sorted(lang_acc.items())]
            )
            log_msg = (
                f"Epoch {epoch:2d}/{args.epochs} | "
                f"LR: {optimizer.param_groups[0]['lr']:.1e} | "
                f"Clips: {train_metrics['grad_clips_pct']:.1f}% | "
                f"Train: {train_metrics['accuracy']:.2%} | "
                f"Test: {eval_metrics['accuracy']:.2%} | {lang_details}"
            )
        else:
            # Show top 5 and bottom 5 for large N
            sorted_langs = sorted(lang_acc.items(), key=lambda x: x[1], reverse=True)
            top5 = sorted_langs[:5]
            bottom5 = sorted_langs[-5:]
            top_str = ", ".join([f"{lang}:{acc:.0%}" for lang, acc in top5])
            bottom_str = ", ".join([f"{lang}:{acc:.0%}" for lang, acc in bottom5])
            log_msg = (
                f"Epoch {epoch:2d}/{args.epochs} | "
                f"LR: {optimizer.param_groups[0]['lr']:.1e} | "
                f"Clips: {train_metrics['grad_clips_pct']:.1f}% | "
                f"Train: {train_metrics['accuracy']:.2%} | "
                f"Test: {eval_metrics['accuracy']:.2%} | "
                f"Top: {top_str}, Bottom: {bottom_str}"
            )

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

    # Save full per-language breakdown
    final_eval = evaluate(model, test_loader, criterion, lang_to_id)
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
