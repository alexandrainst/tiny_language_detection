"""Evaluation script for wavelet spectrogram + CNN model (Phase 3).

Loads a trained wavelet model checkpoint and evaluates it on the test set,
computing accuracy metrics per duration group and per language.

Example:
    uv run src/scripts/evaluate_phase3.py --phase 3 \
        --wavelet-wavelet ricker --wavelet-n-scales 48
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from tiny_language_detection.config.defaults import Config
from tiny_language_detection.data.preprocessing import (
    assign_duration_group,
    get_audio_duration,
    load_and_preprocess,
)
from tiny_language_detection.eval.metrics import compute_metrics
from tiny_language_detection.features.mel_spectrogram import WaveletSpectrogramConfig
from tiny_language_detection.features.mel_spectrogram import (
    extract_wavelet_spectrogram as _extract_wavelet_spectrogram,
)
from tiny_language_detection.models.cnn import LanguageDetectionCNN

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging for the evaluation script."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_manifest: Path,
    data_dir: Path,
    config: Config,
    wavelet_config: WaveletSpectrogramConfig,
    label_map: dict[int, str],
    device: torch.device,
) -> tuple[list[int], list[int], list[str], list[str]]:
    """Run inference on test set with wavelet features.

    Args:
        model:
          Trained model.
        test_manifest:
          Path to test CSV manifest.
        data_dir:
          Directory containing audio files.
        config:
          Audio configuration.
        wavelet_config:
          Wavelet spectrogram configuration for feature extraction.
        label_map:
          Mapping from label indices to language codes.
        device:
          Device to run inference on.

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

        # Extract wavelet spectrogram
        wavelet_spec = _extract_wavelet_spectrogram(
            waveform=waveform, sample_rate=config.sample_rate, config=wavelet_config
        )

        # Get duration group
        if "duration" in df.columns and row["duration"] is not None:
            duration = float(row["duration"])
        else:
            duration = get_audio_duration(
                audio_path=audio_path, target_sr=config.sample_rate
            )
        dur_group = assign_duration_group(duration)

        # Prepare input tensor: [n_scales, time] -> [1, 1, n_scales, time]
        # (batch, channel, n_scales, time) to match the CNN's 4D input.
        wavelet_tensor = (
            torch.from_numpy(wavelet_spec).float().unsqueeze(0).unsqueeze(0).to(device)
        )

        # Forward pass
        outputs = model(wavelet_tensor)
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
        results:
          Dictionary of evaluation metrics.
        output_path:
          Output file path.
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
        description=(
            "Evaluate CNN model for language detection with wavelet features (Phase 3)"
        )
    )
    parser.add_argument(
        "--phase", type=int, default=3, help="Phase number (default: 3)"
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
        "--wavelet-wavelet",
        type=str,
        default="ricker",
        help="Mother wavelet name used in training (default: ricker)",
    )
    parser.add_argument(
        "--wavelet-n-scales",
        type=int,
        default=48,
        help="Number of CWT scales used in training (default: 48)",
    )

    args = parser.parse_args()

    setup_logging()

    # Load training config to get sample rate
    config_path = args.output_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    with open(config_path, "r") as f:
        train_config = json.load(f)

    config = Config(sample_rate=train_config.get("sample_rate", 16000))

    # Wavelet configuration matching training
    wavelet_config = WaveletSpectrogramConfig(
        sample_rate=config.sample_rate,
        widths=args.wavelet_n_scales,
        wavelet=args.wavelet_wavelet,
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
    model = LanguageDetectionCNN(
        num_mfcc=wavelet_config.n_scales,
        time_steps=50,  # Will be overridden by adaptive pooling
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
        wavelet_config=wavelet_config,
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
    setup_logging()
    main()
