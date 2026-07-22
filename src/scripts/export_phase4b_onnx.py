#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "onnx",
#     "onnxscript",
#     "onnxconverter-common",
# ]
# ///
"""Export Phase 4b Compact CNN models to ONNX format for web deployment.

Exports two precision variants:
- FP32: Full precision (~686 KB)
- Float16: Half precision (~343 KB)

Usage:
    uv run src/scripts/export_phase4b_onnx.py
"""

import json
import logging
import sys
from pathlib import Path

import onnx
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_language_detection.models.tiny_cnn import create_small_cnn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_fp32_model(checkpoint_path: Path) -> torch.nn.Module:
    """Load FP32 model from checkpoint.

    Args:
        checkpoint_path:
            Path to PyTorch checkpoint file.

    Returns:
        Model in evaluation mode.
    """
    logger.info(f"Loading FP32 checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = create_small_cnn(num_languages=2)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def export_onnx(
    model: torch.nn.Module,
    output_path: Path,
    precision: str = "float32",
    opset: int = 14,
) -> None:
    """Export model to ONNX format.

    Args:
        model:
            PyTorch model to export.
        output_path:
            Output path for ONNX file.
        precision:
            Precision type (float32 or float16).
        opset:
            ONNX opset version.
    """
    model = model.cpu()
    model.eval()

    # Create dummy input (batch=1, 1 channel, 80 mel bins, 100 time frames)
    dummy_input = torch.randn(1, 1, 80, 100)

    # Dynamic axes for variable-length input
    dynamic_axes = {
        "input": {0: "batch_size", 3: "time_frames"},
        "output": {0: "batch_size"},
    }

    logger.info(f"Exporting {precision} model to {output_path}")

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    # Validate ONNX model
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    # Get file size
    file_size = output_path.stat().st_size
    logger.info(f"Exported {output_path.name}: {file_size / 1024:.1f} KB")

    # Convert precision if needed
    if precision == "float16":
        logger.info("Converting to float16...")
        from onnxconverter_common import float16

        onnx_fp16 = float16.convert_float_to_float16(onnx_model)
        onnx.save(onnx_fp16, str(output_path))
        file_size = output_path.stat().st_size
        logger.info(f"Float16 {output_path.name}: {file_size / 1024:.1f} KB")


def main() -> None:
    """Main export function."""
    # Checkpoint path
    checkpoint_path = Path(
        "data/experiments/phase4b/tiny_cnn_kd/model_best.pth"
    )

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Output directory
    output_dir = Path("web_demo/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and export FP32 model
    model_fp32 = load_fp32_model(checkpoint_path)
    export_onnx(model_fp32, output_dir / "model_fp32.onnx", precision="float32")

    # Export Float16 model
    export_onnx(model_fp32, output_dir / "model_float16.onnx", precision="float16")

    # Create config file for the demo
    config = {
        "models": [
            {
                "name": "FP32 (Full Precision)",
                "file": "model_fp32.onnx",
                "size_kb": round(
                    (output_dir / "model_fp32.onnx").stat().st_size / 1024, 1
                ),
                "precision": "float32",
            },
            {
                "name": "Float16 (Compressed)",
                "file": "model_float16.onnx",
                "size_kb": round(
                    (output_dir / "model_float16.onnx").stat().st_size / 1024, 1
                ),
                "precision": "float16",
            },
        ],
        "input_shape": [1, 1, 80, None],  # Variable time frames
        "sample_rate": 16000,
        "n_mels": 80,
        "labels": ["Danish", "English"],
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config to {config_path}")

    logger.info("\n✅ Export complete! Files ready for web demo:")
    for f in output_dir.iterdir():
        logger.info(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
