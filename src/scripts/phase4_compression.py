#!/usr/bin/env python3
"""Phase 4: Model Compression and Optimisation.

Compresses the Phase 2 (Log-Mel + CNN-RNN) model using various techniques:
- INT8 dynamic quantisation
- FP16 half-precision
- Weight pruning (magnitude-based)
- Combined techniques

Evaluates each variant on the test set and reports accuracy vs size trade-offs.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.utils.prune as prune

# Add src to path for imports
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
from tiny_language_detection.models.cnn_rnn import CNNRNNLanguageDetector

# Constants
TEST_MANIFEST = Path("data/sampled/test.csv")
PHASE2_MODEL = Path("data/experiments/phase2/model.pth")
OUTPUT_DIR = Path("data/experiments/phase4")
DATA_DIR = Path("data")


def setup_logging() -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_phase2_model(
    model_path: Path, device: torch.device
) -> CNNRNNLanguageDetector:
    """Load the Phase 2 baseline model.

    Args:
        model_path: Path to model checkpoint.
        device: Torch device.

    Returns:
        Loaded model in eval mode.
    """
    config_path = model_path.parent / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    model = CNNRNNLanguageDetector(
        n_mels=config["n_mels"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
    )

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model


def get_model_size(model: torch.nn.Module) -> int:
    """Calculate model size in bytes.

    Args:
        model: PyTorch model.

    Returns:
        Size in bytes.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    return param_size + buffer_size


def evaluate_model(
    model: torch.nn.Module,
    device: torch.device,
    phase_name: str,
) -> dict[str, Any]:
    """Evaluate model on test set.

    Args:
        model: Model to evaluate.
        device: Torch device.
        phase_name: Name/version identifier.

    Returns:
        Evaluation metrics dictionary.
    """
    model.eval()
    config = Config()
    mel_config = MelSpectrogramConfig(
        sample_rate=config.sample_rate, n_mels=80, n_fft=400, hop_length=160
    )

    df = pd.read_csv(TEST_MANIFEST)
    predictions = []
    labels = []
    paths_durations = []

    logging.info(f"Evaluating on {len(df)} samples from {TEST_MANIFEST}")

    for idx, row in df.iterrows():
        label = int(row["label"])
        lang = str(row["language"])
        audio_filename = row["path"]

        # Construct audio path
        if lang == "en":
            audio_path = DATA_DIR / "cv26-en" / "clips" / audio_filename
        else:
            audio_path = DATA_DIR / "cv26-da" / audio_filename

        # Load and preprocess audio
        waveform = load_and_preprocess(
            audio_path=audio_path, target_sr=config.sample_rate
        )

        # Extract Log Mel-spectrogram
        log_mel_spec = extract_log_mel_spectrogram(
            waveform=waveform, sample_rate=config.sample_rate, config=mel_config
        )

        # Get duration
        if "duration" in df.columns and row["duration"] is not None:
            duration = float(row["duration"])
        else:
            duration = get_audio_duration(
                audio_path=audio_path, target_sr=config.sample_rate
            )

        # Prepare input tensor (match model dtype)
        log_mel_tensor = torch.from_numpy(log_mel_spec).float().unsqueeze(0).to(device)
        
        # Convert input to model's dtype
        model_dtype = next(model.parameters()).dtype
        if model_dtype == torch.float16:
            log_mel_tensor = log_mel_tensor.half()
        elif model_dtype == torch.float64:
            log_mel_tensor = log_mel_tensor.double()

        # Forward pass
        outputs = model(log_mel_tensor)
        _, predicted = outputs.max(1)
        prediction = predicted.item()

        predictions.append(int(prediction))
        labels.append(int(label))
        paths_durations.append((str(audio_path), duration))

        if (idx + 1) % 100 == 0:
            logging.info(f"  Processed {idx + 1}/{len(df)} samples")

    logging.info("Inference complete: %d samples processed", len(predictions))
    logging.info("Computing metrics...")
    # Build languages list from labels (da=0, en=1)
    languages = ["da" if l == 0 else "en" for l in labels]
    # Build duration groups
    duration_groups = [assign_duration_group(d) for _, d in paths_durations]
    
    metrics = compute_metrics(
        predictions=predictions,
        labels=labels,
        languages=languages,
        duration_groups=duration_groups,
    )

    return metrics


def apply_dynamic_quantisation(model: torch.nn.Module) -> torch.nn.Module:
    """Apply dynamic INT8 quantisation to linear layers.

    Args:
        model: Original model.

    Returns:
        Quantised model.
    """
    return torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.GRU}, dtype=torch.qint8
    )


def apply_fp16_conversion(model: torch.nn.Module) -> torch.nn.Module:
    """Convert model to FP16 half-precision.

    Args:
        model: Original model.

    Returns:
        FP16 model.
    """
    return model.half()


def apply_magnitude_pruning(
    model: torch.nn.Module,
    amount: float = 0.3,
) -> torch.nn.Module:
    """Apply magnitude-based pruning to Conv2d and Linear layers.

    Args:
        model: Original model.
        amount: Fraction of weights to prune (0.0-1.0).

    Returns:
        Pruned model.
    """
    # Prune Conv2d layers
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=amount)
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)

    # Make pruning permanent
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            prune.remove(module, "weight")

    return model


def run_compression_experiments(
    model: torch.nn.Module,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Run all compression experiments.

    Args:
        model: Baseline model.
        test_loader: Test data loader.
        device: Torch device.

    Returns:
        List of experiment results.
    """
    results = []

    # Baseline (FP32)
    logging.info("=" * 60)
    logging.info("Experiment 1: Baseline (FP32)")
    logging.info("=" * 60)
    baseline_size = get_model_size(model)
    baseline_metrics = evaluate_model(model, device, "phase2_baseline_fp32")
    results.append(
        {
            "name": "Baseline (FP32)",
            "method": "none",
            "size_bytes": baseline_size,
            "size_mb": baseline_size / (1024**2),
            "overall_accuracy": baseline_metrics["overall_accuracy"],
            "danish_accuracy": baseline_metrics["per_language_accuracy"]["da"],
            "english_accuracy": baseline_metrics["per_language_accuracy"]["en"],
            "confusion_matrix": baseline_metrics["confusion_matrix"],
        }
    )
    logging.info(f"Size: {baseline_size / (1024**2):.2f} MB")
    logging.info(f"Accuracy: {baseline_metrics['overall_accuracy'] * 100:.2f}%")

    # FP16
    logging.info("\n" + "=" * 60)
    logging.info("Experiment 2: FP16 Half-Precision")
    logging.info("=" * 60)
    model_fp16 = apply_fp16_conversion(model)
    fp16_size = get_model_size(model_fp16)
    fp16_metrics = evaluate_model(
        model_fp16, device, "phase4_fp16"
    )
    results.append(
        {
            "name": "FP16",
            "method": "fp16",
            "size_bytes": fp16_size,
            "size_mb": fp16_size / (1024**2),
            "overall_accuracy": fp16_metrics["overall_accuracy"],
            "danish_accuracy": fp16_metrics["per_language_accuracy"]["da"],
            "english_accuracy": fp16_metrics["per_language_accuracy"]["en"],
            "confusion_matrix": fp16_metrics["confusion_matrix"],
        }
    )
    logging.info(f"Size: {fp16_size / (1024**2):.2f} MB ({fp16_size / baseline_size * 100:.1f}% of baseline)")
    logging.info(f"Accuracy: {fp16_metrics['overall_accuracy'] * 100:.2f}%")

    # INT8 Dynamic Quantisation
    logging.info("\n" + "=" * 60)
    logging.info("Experiment 3: INT8 Dynamic Quantisation")
    logging.info("=" * 60)
    if str(device) == "mps":
        # INT8 quantisation not supported on MPS
        logging.info("Skipping INT8 quantisation (not supported on Apple MPS)")
        results.append(
            {
                "name": "INT8 Dynamic (N/A on MPS)",
                "method": "int8_dynamic",
                "size_bytes": 0,
                "size_mb": 0,
                "overall_accuracy": 0,
                "danish_accuracy": 0,
                "english_accuracy": 0,
                "confusion_matrix": [[0, 0], [0, 0]],
            }
        )
    else:
        model_int8_dyn = apply_dynamic_quantisation(model)
        int8_dyn_size = get_model_size(model_int8_dyn)
        int8_dyn_metrics = evaluate_model(
            model_int8_dyn, device, "phase4_int8_dynamic"
        )
        results.append(
            {
                "name": "INT8 Dynamic",
                "method": "int8_dynamic",
                "size_bytes": int8_dyn_size,
                "size_mb": int8_dyn_size / (1024**2),
                "overall_accuracy": int8_dyn_metrics["overall_accuracy"],
                "danish_accuracy": int8_dyn_metrics["per_language_accuracy"]["da"],
                "english_accuracy": int8_dyn_metrics["per_language_accuracy"]["en"],
                "confusion_matrix": int8_dyn_metrics["confusion_matrix"],
            }
        )
        logging.info(f"Size: {int8_dyn_size / (1024**2):.2f} MB ({int8_dyn_size / baseline_size * 100:.1f}% of baseline)")
        logging.info(f"Accuracy: {int8_dyn_metrics['overall_accuracy'] * 100:.2f}%")

    # Pruning 20%
    logging.info("\n" + "=" * 60)
    logging.info("Experiment 4: Magnitude Pruning (20%)")
    logging.info("=" * 60)
    model_prune20 = apply_magnitude_pruning(model, amount=0.2)
    prune20_size = get_model_size(model_prune20)
    prune20_metrics = evaluate_model(
        model_prune20, device, "phase4_prune20"
    )
    results.append(
        {
            "name": "Pruning 20%",
            "method": "pruning_20",
            "size_bytes": prune20_size,
            "size_mb": prune20_size / (1024**2),
            "overall_accuracy": prune20_metrics["overall_accuracy"],
            "danish_accuracy": prune20_metrics["per_language_accuracy"]["da"],
            "english_accuracy": prune20_metrics["per_language_accuracy"]["en"],
            "confusion_matrix": prune20_metrics["confusion_matrix"],
        }
    )
    logging.info(f"Size: {prune20_size / (1024**2):.2f} MB ({prune20_size / baseline_size * 100:.1f}% of baseline)")
    logging.info(f"Accuracy: {prune20_metrics['overall_accuracy'] * 100:.2f}%")

    # Pruning 40%
    logging.info("\n" + "=" * 60)
    logging.info("Experiment 5: Magnitude Pruning (40%)")
    logging.info("=" * 60)
    model_prune40 = apply_magnitude_pruning(model, amount=0.4)
    prune40_size = get_model_size(model_prune40)
    prune40_metrics = evaluate_model(
        model_prune40, device, "phase4_prune40"
    )
    results.append(
        {
            "name": "Pruning 40%",
            "method": "pruning_40",
            "size_bytes": prune40_size,
            "size_mb": prune40_size / (1024**2),
            "overall_accuracy": prune40_metrics["overall_accuracy"],
            "danish_accuracy": prune40_metrics["per_language_accuracy"]["da"],
            "english_accuracy": prune40_metrics["per_language_accuracy"]["en"],
            "confusion_matrix": prune40_metrics["confusion_matrix"],
        }
    )
    logging.info(f"Size: {prune40_size / (1024**2):.2f} MB ({prune40_size / baseline_size * 100:.1f}% of baseline)")
    logging.info(f"Accuracy: {prune40_metrics['overall_accuracy'] * 100:.2f}%")

    # Pruning 60%
    logging.info("\n" + "=" * 60)
    logging.info("Experiment 6: Magnitude Pruning (60%)")
    logging.info("=" * 60)
    model_prune60 = apply_magnitude_pruning(model, amount=0.6)
    prune60_size = get_model_size(model_prune60)
    prune60_metrics = evaluate_model(
        model_prune60, device, "phase4_prune60"
    )
    results.append(
        {
            "name": "Pruning 60%",
            "method": "pruning_60",
            "size_bytes": prune60_size,
            "size_mb": prune60_size / (1024**2),
            "overall_accuracy": prune60_metrics["overall_accuracy"],
            "danish_accuracy": prune60_metrics["per_language_accuracy"]["da"],
            "english_accuracy": prune60_metrics["per_language_accuracy"]["en"],
            "confusion_matrix": prune60_metrics["confusion_matrix"],
        }
    )
    logging.info(f"Size: {prune60_size / (1024**2):.2f} MB ({prune60_size / baseline_size * 100:.1f}% of baseline)")
    logging.info(f"Accuracy: {prune60_metrics['overall_accuracy'] * 100:.2f}%")

    # Combined: FP16 + Pruning 20%
    logging.info("\n" + "=" * 60)
    logging.info("Experiment 7: FP16 + Pruning 20%")
    logging.info("=" * 60)
    model_combined = apply_fp16_conversion(apply_magnitude_pruning(model, amount=0.2))
    combined_size = get_model_size(model_combined)
    combined_metrics = evaluate_model(
        model_combined, device, "phase4_fp16_prune20"
    )
    results.append(
        {
            "name": "FP16 + Pruning 20%",
            "method": "fp16_prune20",
            "size_bytes": combined_size,
            "size_mb": combined_size / (1024**2),
            "overall_accuracy": combined_metrics["overall_accuracy"],
            "danish_accuracy": combined_metrics["per_language_accuracy"]["da"],
            "english_accuracy": combined_metrics["per_language_accuracy"]["en"],
            "confusion_matrix": combined_metrics["confusion_matrix"],
        }
    )
    logging.info(f"Size: {combined_size / (1024**2):.2f} MB ({combined_size / baseline_size * 100:.1f}% of baseline)")
    logging.info(f"Accuracy: {combined_metrics['overall_accuracy'] * 100:.2f}%")

    return results


def save_results(results: list[dict[str, Any]], output_path: Path) -> None:
    """Save compression experiment results.

    Args:
        results: List of experiment results.
        output_path: Output JSON path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved results to {output_path}")


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print summary table of all experiments.

    Args:
        results: List of experiment results.
    """
    print("\n" + "=" * 80)
    print("PHASE 4 COMPRESSION SUMMARY")
    print("=" * 80)
    print(f"{'Method':<25} {'Size (MB)':<12} {'Size Ratio':<12} {'Accuracy':<12} {'Δ Acc':<10}")
    print("-" * 80)

    baseline_acc = results[0]["overall_accuracy"]
    baseline_size = results[0]["size_mb"]

    for r in results:
        size_ratio = r["size_mb"] / baseline_size * 100
        acc_pct = r["overall_accuracy"] * 100
        delta_acc = (r["overall_accuracy"] - baseline_acc) * 100
        print(f"{r['name']:<25} {r['size_mb']:<12.3f} {size_ratio:<12.1f} {acc_pct:<12.2f} {delta_acc:+.2f}")

    print("=" * 80)
    print("\nSize targets:")
    print(f"  Earbuds: <1 MB")
    print(f"  Headphones: 2-3 MB")
    print("\nBest methods by criterion:")

    # Best accuracy (excluding baseline)
    best_acc = max(results[1:], key=lambda x: x["overall_accuracy"])
    print(f"  Best accuracy: {best_acc['name']} ({best_acc['overall_accuracy'] * 100:.2f}%)")

    # Smallest under 1 MB
    under_1mb = [r for r in results if r["size_mb"] < 1.0]
    if under_1mb:
        best_small = max(under_1mb, key=lambda x: x["overall_accuracy"])
        print(f"  Best under 1 MB: {best_small['name']} ({best_small['overall_accuracy'] * 100:.2f}%, {best_small['size_mb']:.3f} MB)")

    # Best under 2% accuracy loss
    acceptable = [r for r in results if (baseline_acc - r["overall_accuracy"]) * 100 <= 2.0]
    if acceptable:
        best_efficient = min(acceptable, key=lambda x: x["size_mb"])
        print(f"  Best under 2% loss: {best_efficient['name']} ({best_efficient['overall_accuracy'] * 100:.2f}%, {best_efficient['size_mb']:.3f} MB)")

    print("=" * 80)


def main() -> None:
    """Main entry point."""
    setup_logging()

    # Check for GPU/MPS
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load baseline model
    logging.info(f"Loading Phase 2 model from {PHASE2_MODEL}")
    model = load_phase2_model(PHASE2_MODEL, device)
    logging.info(f"Model parameters: {model.count_parameters():,}")

    # Run compression experiments
    logging.info("Starting compression experiments...")
    results = run_compression_experiments(model, device)

    # Save results
    results_path = OUTPUT_DIR / "compression_results.json"
    save_results(results, results_path)

    # Print summary
    print_summary(results)

    logging.info(f"All results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
