"""Helper script to run Phase 3 (wavelet features) end-to-end.

This script trains a wavelet spectrogram + CNN model and evaluates it on the test set,
automating the full pipeline from training through evaluation.

Example:
    uv run src/scripts/run_phase3.py --train-epochs 30 \
        --eval-wavelet ricker --eval-n-scales 48
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def run_command(cmd: list[str], description: str) -> None:
    """Run a shell command and log progress.

    Args:
        cmd:
          Command to execute as a list of arguments.
        description:
          Human-readable description for logging.

    Raises:
        RuntimeError:
          If the command exits with a non-zero status code.
    """
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{description} failed with exit code {result.returncode}")
    logger.info("%s complete", description)


def main() -> None:
    """Run Phase 3 training and evaluation end-to-end."""
    parser = argparse.ArgumentParser(
        description="Run Phase 3 wavelet features pipeline end-to-end"
    )
    parser.add_argument(
        "--train-epochs", type=int, default=30, help="Training epochs (default: 30)"
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)",
    )
    parser.add_argument(
        "--train-lr",
        type=float,
        default=0.001,
        help="Training learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--train-wavelet",
        type=str,
        default="ricker",
        help="Training wavelet name (default: ricker)",
    )
    parser.add_argument(
        "--train-n-scales",
        type=int,
        default=48,
        help="Training number of CWT scales (default: 48)",
    )
    parser.add_argument(
        "--eval-wavelet",
        type=str,
        default="ricker",
        help="Evaluation wavelet name (default: ricker)",
    )
    parser.add_argument(
        "--eval-n-scales",
        type=int,
        default=48,
        help="Evaluation number of CWT scales (default: 48)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory (default: data)",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger.info("=" * 60)
    logger.info("Phase 3: Wavelet Spectrogram + CNN Pipeline")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Train wavelet: {args.train_wavelet}")
    logger.info(f"  Train n_scales: {args.train_n_scales}")
    logger.info(f"  Eval wavelet: {args.eval_wavelet}")
    logger.info(f"  Eval n_scales: {args.eval_n_scales}")
    logger.info("")

    # Determine project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    # Step 1: Train the model
    logger.info("Step 1: Training wavelet model...")
    logger.info("-" * 40)
    train_cmd = [
        sys.executable,
        str(script_dir / "train_phase3.py"),
        "--phase",
        "3",
        "--epochs",
        str(args.train_epochs),
        "--batch-size",
        str(args.train_batch_size),
        "--lr",
        str(args.train_lr),
        "--wavelet-wavelet",
        args.train_wavelet,
        "--wavelet-n-scales",
        str(args.train_n_scales),
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(project_root / "data" / "experiments" / "phase3"),
    ]
    run_command(train_cmd, "Training")

    # Step 2: Evaluate the model
    logger.info("")
    logger.info("Step 2: Evaluating wavelet model...")
    logger.info("-" * 40)
    eval_cmd = [
        sys.executable,
        str(script_dir / "evaluate_phase3.py"),
        "--phase",
        "3",
        "--wavelet-wavelet",
        args.eval_wavelet,
        "--wavelet-n-scales",
        str(args.eval_n_scales),
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(project_root / "data" / "experiments" / "phase3"),
    ]
    run_command(eval_cmd, "Evaluation")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 3 pipeline complete!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {project_root / 'data' / 'experiments' / 'phase3'}")
    logger.info("")
    logger.info("To view results:")
    logger.info(
        "  cat %s/evaluation.json",
        project_root / "data" / "experiments" / "phase3" / "evaluation.json",
    )
    logger.info(
        "  cat %s/training_history.json",
        project_root / "data" / "experiments" / "phase3" / "training_history.json",
    )


if __name__ == "__main__":
    main()
