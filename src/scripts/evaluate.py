#!/usr/bin/env -S uv run
"""Evaluation script for Phase 1: MFCC + CNN baseline.

Evaluates a trained model on the test set and computes metrics.

Usage:
    uv run src/scripts/evaluate.py --phase 1
        [--checkpoint data/experiments/phase1/model.pth]
"""

import argparse
import json
import logging
import sys
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
from tiny_language_detection.features.mfcc import extract_mfcc
from tiny_language_detection.models import LanguageDetectionCNN

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_checkpoint(
    checkpoint_path: Path, config_path: Path
) -> tuple[LanguageDetectionCNN, dict, dict[str, int]]:
    """Load model checkpoint and configuration.

    Args:
        checkpoint_path:
            Path to model.pth checkpoint file.
        config_path:
            Path to config.json file.

    Returns:
        Tuple of (model, training_config, label_map).

    Raises:
        FileNotFoundError:
            If checkpoint or config files are missing.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Please train a model first or specify the correct checkpoint path."
        )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found at {config_path}. "
            "Please ensure the experiment directory contains config.json."
        )

    # Load training config
    with open(config_path) as f:
        training_config = json.load(f)

    # Load label map
    label_map_path = checkpoint_path.parent / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError(
            f"Label map not found at {label_map_path}. "
            "Please ensure the experiment directory contains label_map.json."
        )

    with open(label_map_path) as f:
        label_map = json.load(f)

    # Create model with saved dimensions
    num_mfcc = training_config.get("num_mfcc", 40)
    num_languages = training_config.get("num_languages", 2)
    # Use a reasonable default for time_steps if not saved
    time_steps = training_config.get("time_steps", 50)

    model = LanguageDetectionCNN(
        num_mfcc=num_mfcc, time_steps=time_steps, num_languages=num_languages
    )

    # Load weights
    state_dict = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(state_dict)

    return model, training_config, label_map


def load_test_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load test manifest from CSV file.

    Args:
        manifest_path:
            Path to test.csv manifest file.

    Returns:
        DataFrame with test samples.

    Raises:
        FileNotFoundError:
            If manifest file is missing.
        ValueError:
            If manifest is missing required columns.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Test manifest not found at {manifest_path}. "
            "Please run the data sampling script first to create test.csv."
        )

    df = pd.read_csv(manifest_path)

    required_columns = {"path", "language"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Test manifest missing required columns: {missing_columns}. "
            "Expected columns: path, language, and optionally duration, duration_group."
        )

    return df


def run_inference(
    model: LanguageDetectionCNN,
    manifest_df: pd.DataFrame,
    config: Config,
    label_map: dict[str, int],
) -> tuple[list[int], list[int], list[str], list[str]]:
    """Run inference on test samples.

    Args:
        model:
            Trained model for prediction.
        manifest_df:
            DataFrame with test samples.
        config:
            Configuration object.
        label_map:
            Mapping from language codes to labels.

    Returns:
        Tuple of (predictions, labels, languages, duration_groups).
    """
    # Create reverse label map
    {v: k for k, v in label_map.items()}

    predictions: list[int] = []
    labels: list[int] = []
    languages: list[str] = []
    duration_groups: list[str] = []

    model.eval()
    device = next(model.parameters()).device

    logger.info("Running inference on %d test samples...", len(manifest_df))

    for idx, row in manifest_df.iterrows():
        audio_path = Path(row["path"])
        language = str(row["language"])

        # Skip if audio file doesn't exist
        if not audio_path.exists():
            logger.warning(
                "Audio file not found: %s (skipping sample %d)", audio_path, idx
            )
            continue

        # Get label
        label = label_map.get(language, -1)
        if label == -1:
            logger.warning(
                "Unknown language '%s' for sample %d (skipping)", language, idx
            )
            continue

        # Load and preprocess audio
        waveform = load_and_preprocess(
            audio_path=audio_path, target_sr=config.sample_rate
        )

        # Extract MFCC features
        mfccs = extract_mfcc(
            waveform=waveform, sample_rate=config.sample_rate, config=config
        )

        # Prepare input tensor [batch=1, num_mfcc, time_steps, 1]
        # mfccs shape: [num_mfcc, time_steps]
        input_tensor = mfccs.unsqueeze(0).unsqueeze(-1)
        input_tensor = input_tensor.to(device)

        # Forward pass
        with torch.no_grad():
            logits = model(input_tensor)
            prediction = torch.argmax(logits, dim=1).item()

        # Get duration group
        if "duration_group" in manifest_df.columns:
            dur_group = str(row["duration_group"])
        elif "duration" in manifest_df.columns:
            dur_group = assign_duration_group(float(row["duration"]))
        else:
            # Compute duration from audio file
            duration = get_audio_duration(
                audio_path=audio_path, target_sr=config.sample_rate
            )
            dur_group = assign_duration_group(duration)

        predictions.append(int(prediction))
        labels.append(int(label))
        languages.append(language)
        duration_groups.append(dur_group)

    logger.info("Inference complete: %d samples processed", len(predictions))

    return predictions, labels, languages, duration_groups


def save_results(results: dict, output_path: Path) -> None:
    """Save evaluation results to JSON file.

    Args:
        results:
            Dictionary of evaluation metrics.
        output_path:
            Path to save JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_path)


def print_summary(results: dict) -> None:
    """Print evaluation summary to console.

    Args:
        results:
            Dictionary of evaluation metrics.
    """
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nTotal samples: {results['total_samples']}")
    print(f"Overall accuracy: {results['overall_accuracy']:.2%}")

    print("\nAccuracy by duration group:")
    for group, acc in sorted(results["accuracy_by_duration"].items()):
        print(f"  {group}: {acc:.2%}")

    print("\nPer-language accuracy:")
    for lang, acc in sorted(results["per_language_accuracy"].items()):
        print(f"  {lang}: {acc:.2%}")

    print("\nConfusion matrix:")
    cm = results["confusion_matrix"]
    print(f"  [[{cm[0][0]}, {cm[0][1]}],")
    print(f"   [{cm[1][0]}, {cm[1][1]}]]")
    print("   (rows: true labels, columns: predicted labels)")

    print("\n" + "=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate MFCC+CNN model for language detection"
    )
    parser.add_argument(
        "--phase",
        type=int,
        required=True,
        help="Phase number (e.g., 1 for MFCC+CNN baseline)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: data/experiments/phaseN/model.pth)",
    )
    parser.add_argument(
        "--test-manifest",
        type=str,
        default=None,
        help="Path to test manifest (default: data/sampled/test.csv)",
    )
    return parser.parse_args()


def main() -> None:
    """Main evaluation entry point."""
    args = parse_args()

    # Determine paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    # Use provided paths or defaults
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = (
            project_root / "data" / "experiments" / f"phase{args.phase}" / "model.pth"
        )

    if args.test_manifest:
        manifest_path = Path(args.test_manifest)
    else:
        manifest_path = project_root / "data" / "sampled" / "test.csv"

    config_path = checkpoint_path.parent / "config.json"
    output_path = checkpoint_path.parent / "evaluation.json"

    logger.info("=" * 60)
    logger.info("Phase %d Evaluation", args.phase)
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info(f"  Checkpoint: {checkpoint_path}")
    logger.info(f"  Test manifest: {manifest_path}")
    logger.info(f"  Output: {output_path}")
    logger.info("=" * 60)

    # Load config
    config = Config()

    # Load checkpoint and model
    try:
        model, training_config, label_map = load_checkpoint(
            checkpoint_path=checkpoint_path, config_path=config_path
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("Loaded model from %s", checkpoint_path)
    logger.info("Using device: %s", device)

    # Load test manifest
    try:
        manifest_df = load_test_manifest(manifest_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except ValueError as e:
        logger.error("Invalid manifest format: %s", e)
        sys.exit(1)

    logger.info("Loaded %d test samples from %s", len(manifest_df), manifest_path)

    # Run inference
    predictions, labels, languages, duration_groups = run_inference(
        model=model, manifest_df=manifest_df, config=config, label_map=label_map
    )

    if len(predictions) == 0:
        logger.error("No samples were processed. Check audio file paths.")
        sys.exit(1)

    # Compute metrics
    results = compute_metrics(
        predictions=predictions,
        labels=labels,
        languages=languages,
        duration_groups=duration_groups,
    )

    # Add training metadata
    results["phase"] = args.phase
    results["checkpoint"] = str(checkpoint_path)
    results["test_manifest"] = str(manifest_path)

    # Save results
    save_results(results, output_path)

    # Print summary
    print_summary(results)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
