#!/usr/bin/env -S uv run
"""Evaluate CNN-RNN model with Log Mel-spectrogram features (Phase 2).

Uses the same test set as Phase 1 for direct comparison.

Usage:
    uv run src/scripts/evaluate_phase2.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch

from tiny_language_detection.config import Config
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
from tiny_language_detection.models import CNNRNNLanguageDetector


def setup_logging() -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    test_manifest: Path,
    data_dir: Path,
    config: Config,
    mel_config: MelSpectrogramConfig,
    label_map: dict[int, str],
    device: torch.device,
) -> tuple[list[int], list[int], list[str], list[str]]:
    """Run inference on test set.

    Args:
        model: Trained model.
        test_manifest: Path to test CSV manifest.
        data_dir: Directory containing audio files.
        config: Audio configuration.
        mel_config: Mel-spectrogram configuration.
        label_map: Mapping from label indices to language codes.
        device: Device to run inference on.

    Returns:
        Tuple of (predictions, labels, languages, duration_groups).
    """
    model.eval()

    df = pd.read_csv(test_manifest)
    predictions = []
    labels = []
    languages = []
    duration_groups = []

    logger.info(f"Evaluating on {len(df)} samples from {test_manifest}")

    for idx, row in df.iterrows():
        label = int(row["label"])
        str(row["language"])

        # Construct audio path
        lang = str(row["language"])
        audio_filename = row["path"]

        if lang == "en":
            audio_path = data_dir / "cv26-en" / "clips" / audio_filename
        else:
            audio_path = data_dir / "cv26-da" / audio_filename

        # Load and preprocess audio (returns waveform at target_sr)
        waveform = load_and_preprocess(
            audio_path=audio_path, target_sr=config.sample_rate
        )

        # Extract Log Mel-spectrogram
        log_mel_spec = extract_log_mel_spectrogram(
            waveform=waveform, sample_rate=config.sample_rate, config=mel_config
        )

        # Get duration group
        if "duration" in df.columns and row["duration"] is not None:
            duration = float(row["duration"])
        else:
            duration = get_audio_duration(
                audio_path=audio_path, target_sr=config.sample_rate
            )
        dur_group = assign_duration_group(duration)

        # Prepare input tensor
        log_mel_tensor = torch.from_numpy(log_mel_spec).float().unsqueeze(0).to(device)

        # Forward pass
        outputs = model(log_mel_tensor)
        _, predicted = outputs.max(1)
        prediction = predicted.item()

        predictions.append(int(prediction))
        labels.append(int(label))
        languages.append(lang)
        duration_groups.append(dur_group)

    logger.info("Inference complete: %d samples processed", len(predictions))

    return predictions, labels, languages, duration_groups


def save_results(results: dict, output_path: Path) -> None:
    """Save evaluation results to JSON.

    Args:
        results: Dictionary of evaluation metrics.
        output_path: Output file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {output_path}")


def main() -> None:
    """Main evaluation entry point.

    Raises:
        FileNotFoundError:
            If the model checkpoint does not exist.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate CNN-RNN model for language detection (Phase 2)"
    )
    parser.add_argument(
        "--phase", type=int, default=2, help="Phase number (default: 2)"
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
        "--n-mels",
        type=int,
        default=80,
        help="Number of Mel bins used in training (default: 80)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="GRU hidden size used in training (default: 64)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="Number of GRU layers used in training (default: 1)",
    )

    args = parser.parse_args()

    setup_logging()

    # Configuration
    config = Config()
    mel_config = MelSpectrogramConfig(
        sample_rate=config.sample_rate, n_mels=args.n_mels, n_fft=400, hop_length=160
    )

    # Load label map
    label_map_path = args.output_dir / "label_map.json"
    with open(label_map_path, "r") as f:
        label_to_lang = json.load(f)
    label_map = {int(k): v for k, v in label_to_lang.items()}

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    num_languages = len(label_map)
    model = CNNRNNLanguageDetector(
        n_mels=args.n_mels,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_languages=num_languages,
    ).to(device)

    model_path = args.output_dir / "model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model.load_state_dict(torch.load(model_path, weights_only=True))
    logger.info(f"Loaded model from {model_path}")

    # Evaluate
    test_manifest = args.data_dir / "sampled" / "test.csv"
    predictions, labels, languages, duration_groups = evaluate(
        model=model,
        test_manifest=test_manifest,
        data_dir=args.data_dir,
        config=config,
        mel_config=mel_config,
        label_map=label_map,
        device=device,
    )

    # Compute metrics
    results = compute_metrics(
        predictions=predictions,
        labels=labels,
        languages=languages,
        duration_groups=duration_groups,
    )
    results["phase"] = args.phase
    results["checkpoint"] = str(model_path.absolute())
    results["test_manifest"] = str(test_manifest.absolute())

    # Save results
    results_path = args.output_dir / "evaluation.json"
    save_results(results, results_path)

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info("")
    logger.info(f"Total samples: {results['total_samples']}")
    logger.info(f"Overall accuracy: {results['overall_accuracy']:.2%}")
    logger.info("")
    logger.info("Accuracy by duration group:")
    for group, acc in sorted(results["accuracy_by_duration"].items()):
        logger.info(f"  {group}: {acc:.2%}")
    logger.info("")
    logger.info("Per-language accuracy:")
    for lang, acc in sorted(results["per_language_accuracy"].items()):
        logger.info(f"  {lang}: {acc:.2%}")
    logger.info("")
    logger.info("Confusion matrix:")
    cm = results["confusion_matrix"]
    logger.info(f"  [[{cm[0][0]}, {cm[0][1]}],")
    logger.info(f"   [{cm[1][0]}, {cm[1][1]}]]")
    logger.info("   (rows: true labels, columns: predicted labels)")
    logger.info("")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
