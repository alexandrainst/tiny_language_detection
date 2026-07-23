#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "onnx",
#     "onnxconverter-common",
#     "onnxscript",
# ]
# ///
"""Export Phase 4b model to ONNX for web demo.

Exports FP16 model only - best balance of size and accuracy for web deployment.

Usage:
    uv run src/scripts/export_web_demo.py
"""

import json
import logging
import sys
from pathlib import Path

import onnx
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_language_detection.models.tiny_cnn import create_small_cnn

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main export function."""
    checkpoint_path = Path("data/experiments/phase4b/tiny_cnn_kd/model_best.pth")

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Output directory
    output_dir = Path("web_demo/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = create_small_cnn(num_languages=2)
    model.load_state_dict(checkpoint)
    model.eval()

    # Export to ONNX
    output_path = output_dir / "model.onnx"
    dummy_input = torch.randn(1, 1, 80, 100)

    logger.info(f"Exporting FP32 model to {output_path}")

    # Export with older API to avoid external data
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 3: "time_frames"},
            "output": {0: "batch_size"},
        },
    )

    # Validate
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    file_size = output_path.stat().st_size
    logger.info(f"Exported {output_path.name}: {file_size / 1024:.1f} KB")

    # Create config
    config = {
        "models": [
            {
                "name": "Model",
                "file": "model.onnx",
                "disk_kb": round(file_size / 1024),
                "ram_kb": 886,
                "precision": "float32",
                "accuracy": "96.65%",
            }
        ],
        "input_shape": [1, 1, 80, None],
        "sample_rate": 16000,
        "n_mels": 80,
        "labels": ["Danish", "English"],
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config to {config_path}")

    logger.info("\n✅ Export complete!")
    logger.info(f"  Model: {output_path.name} ({file_size / 1024:.1f} KB)")
    logger.info("  RAM: ~543 KB at runtime")
    logger.info("  Accuracy: 96.65%")


if __name__ == "__main__":
    main()
