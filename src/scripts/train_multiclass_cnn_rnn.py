#!/usr/bin/env python3
"""Train CNN-RNN for multi-class language detection (N classes).

Supports:
- Binary classification (e.g., Danish vs English)
- Multi-class classification (e.g., 23 languages from YODAS-Granary)
- Custom N-language datasets from HuggingFace or local manifests
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tiny_language_detection.data.multiclass_dataset import (
    MulticlassDataset,
    collate_fn,
)
from tiny_language_detection.features.mel_spectrogram import MelSpectrogramConfig
from tiny_language_detection.models.cnn_rnn import CNNRNNLanguageDetector
from tiny_language_detection.training import (
    DEVICE,
    compute_class_weights,
    evaluate,
    save_best_model,
    save_checkpoint,
    train_epoch,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train CNN-RNN for Multi-Class Language Detection"
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    # SpecAugment parameters
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

    # Feature extraction parameters
    parser.add_argument(
        "--n-mels",
        type=int,
        default=80,
        help="Number of Mel filter banks (default: 80)",
    )
    parser.add_argument(
        "--n-fft", type=int, default=400, help="FFT window size (default: 400)"
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

    # Model architecture
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=[32, 64, 128],
        metavar="C",
        help="CNN channels per block (default: 32 64 128)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="GRU hidden layer size (default: 64)",
    )
    parser.add_argument(
        "--num-layers", type=int, default=1, help="Number of GRU layers (default: 1)"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.3, help="Dropout probability (default: 0.3)"
    )
    parser.add_argument(
        "--num-languages",
        type=int,
        default=None,
        help="Number of output classes (auto-detected from dataset if not specified)",
    )

    # Data and output
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
        help="Output dir for checkpoints (default: data/experiments/{dataset_name})",
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

    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Class weights enabled by default for multi-class (N > 2)
    use_class_weights = not args.no_class_weights

    # Create default output directory based on dataset name
    if args.output_dir is None:
        dataset_name = (
            Path(args.dataset).name if args.use_hf else Path(args.dataset).stem
        )
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
        f"({'auto-detected' if args.num_languages is None else 'specified'})"
    )
    logger.info(f"  Class weights: {use_class_weights}")
    logger.info(f"  CNN channels: {args.channels} ({len(args.channels)} blocks)")
    logger.info(f"  GRU hidden size: {args.hidden_size}")
    logger.info(f"  GRU layers: {args.num_layers}")
    logger.info(f"  Dropout: {args.dropout}")
    logger.info(
        f"  Mel: n_mels={args.n_mels}, n_fft={args.n_fft}, hop={args.hop_length}"
    )
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
    model = CNNRNNLanguageDetector(
        n_mels=mel_config.n_mels,
        channels=args.channels,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_languages=num_classes,
        dropout=args.dropout,
    )

    logger.info(f"Model parameters: {model.count_parameters():,}")

    model.to(DEVICE)

    # Class weights for imbalanced datasets
    if use_class_weights:
        class_weights = compute_class_weights(train_dataset.labels)
        class_weights_tensor = class_weights.to(DEVICE)
        weight_range = f"{class_weights.min():.2f}-{class_weights.max():.2f}"
        logger.info(f"Class weights (range: {weight_range})")
    else:
        class_weights_tensor = None

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_accuracy = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, args.max_grad_norm
        )
        eval_metrics = evaluate(
            model, test_loader, criterion, list(train_dataset.lang_to_id.keys())
        )

        scheduler.step()

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "args": vars(args),
        }
        save_checkpoint(
            checkpoint, args.output_dir, f"checkpoint_epoch_{epoch}.pth.tar"
        )

        # Log progress
        if num_classes <= 5:
            # Show all languages for small N
            lang_acc = eval_metrics.get("lang_accuracy", {})
            lang_details = ", ".join(
                [f"{lang}:{acc:.0%}" for lang, acc in sorted(lang_acc.items())]
            )
        else:
            # Show top 5 and bottom 5 for large N
            lang_acc = eval_metrics.get("lang_accuracy", {})
            sorted_langs = sorted(lang_acc.items(), key=lambda x: x[1], reverse=True)
            top5 = sorted_langs[:5]
            bottom5 = sorted_langs[-5:]
            top_str = ", ".join([f"{lang}:{acc:.0%}" for lang, acc in top5])
            bottom_str = ", ".join([f"{lang}:{acc:.0%}" for lang, acc in bottom5])
            lang_details = f"Top: {top_str}, Bottom: {bottom_str}"

        logger.info(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"LR: {optimizer.param_groups[0]['lr']:.1e} | "
            f"Train: {train_metrics['accuracy']:.2%} | "
            f"Test: {eval_metrics['accuracy']:.2%} | {lang_details}"
        )

        # Save best model
        if eval_metrics["accuracy"] > best_accuracy:
            best_accuracy = eval_metrics["accuracy"]
            best_epoch = epoch
            save_best_model(model, mel_config, args.output_dir)

    logger.info("=" * 60)
    logger.info(f"Training complete! Best: {best_accuracy:.2%} at epoch {best_epoch}")
    logger.info(f"Checkpoints saved to: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
