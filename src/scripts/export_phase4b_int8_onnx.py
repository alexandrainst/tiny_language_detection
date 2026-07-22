#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "onnx",
#     "onnxscript",
# ]
# ///
"""Export INT8 model with embedded dequantisation for web deployment.

This creates an ONNX model where:
- INT8 weights are stored as constants (saving disk + RAM)
- Dequantisation happens via ONNX operators in the graph
- Runtime memory stays at INT8 size until dequantisation

Usage:
    uv run src/scripts/export_phase4b_int8_onnx.py
"""

import logging
import sys
from pathlib import Path

import onnx
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_language_detection.models.tiny_cnn import create_small_cnn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class INT8CompactCNN(nn.Module):
    """Compact CNN with INT8 weights and on-the-fly dequantisation."""

    def __init__(self, quantised_checkpoint: dict):
        super().__init__()
        compressed = quantised_checkpoint["model"]
        metadata = quantised_checkpoint["metadata"]

        # Store INT8 weights and FP32 biases
        # Conv layers
        self.conv1_weight = nn.Parameter(
            compressed["features.0.weight"].to(torch.int8), requires_grad=False
        )
        self.conv1_bias = nn.Parameter(
            compressed["features.0.bias"], requires_grad=False
        )
        self.conv1_scale = torch.tensor(metadata["scales"]["features.0.weight"])
        self.conv1_zero = torch.tensor(metadata["zeros"]["features.0.weight"])

        self.bn1_weight = nn.Parameter(compressed["features.1.weight"], requires_grad=False)
        self.bn1_bias = nn.Parameter(compressed["features.1.bias"], requires_grad=False)
        self.bn1_mean = nn.Parameter(
            compressed["features.1.running_mean"], requires_grad=False
        )
        self.bn1_var = nn.Parameter(
            compressed["features.1.running_var"], requires_grad=False
        )

        self.conv2_weight = nn.Parameter(
            compressed["features.4.weight"].to(torch.int8), requires_grad=False
        )
        self.conv2_bias = nn.Parameter(
            compressed["features.4.bias"], requires_grad=False
        )
        self.conv2_scale = torch.tensor(metadata["scales"]["features.4.weight"])
        self.conv2_zero = torch.tensor(metadata["zeros"]["features.4.weight"])

        self.bn2_weight = nn.Parameter(compressed["features.5.weight"], requires_grad=False)
        self.bn2_bias = nn.Parameter(compressed["features.5.bias"], requires_grad=False)
        self.bn2_mean = nn.Parameter(
            compressed["features.5.running_mean"], requires_grad=False
        )
        self.bn2_var = nn.Parameter(
            compressed["features.5.running_var"], requires_grad=False
        )

        self.conv3_weight = nn.Parameter(
            compressed["features.8.weight"].to(torch.int8), requires_grad=False
        )
        self.conv3_bias = nn.Parameter(
            compressed["features.8.bias"], requires_grad=False
        )
        self.conv3_scale = torch.tensor(metadata["scales"]["features.8.weight"])
        self.conv3_zero = torch.tensor(metadata["zeros"]["features.8.weight"])

        self.bn3_weight = nn.Parameter(compressed["features.9.weight"], requires_grad=False)
        self.bn3_bias = nn.Parameter(compressed["features.9.bias"], requires_grad=False)
        self.bn3_mean = nn.Parameter(
            compressed["features.9.running_mean"], requires_grad=False
        )
        self.bn3_var = nn.Parameter(
            compressed["features.9.running_var"], requires_grad=False
        )

        # FC layers
        self.fc1_weight = nn.Parameter(
            compressed["classifier.0.weight"].to(torch.int8), requires_grad=False
        )
        self.fc1_bias = nn.Parameter(
            compressed["classifier.0.bias"], requires_grad=False
        )
        self.fc1_scale = torch.tensor(metadata["scales"]["classifier.0.weight"])
        self.fc1_zero = torch.tensor(metadata["zeros"]["classifier.0.weight"])

        self.fc2_weight = nn.Parameter(
            compressed["classifier.3.weight"].to(torch.int8), requires_grad=False
        )
        self.fc2_bias = nn.Parameter(
            compressed["classifier.3.bias"], requires_grad=False
        )
        self.fc2_scale = torch.tensor(metadata["scales"]["classifier.3.weight"])
        self.fc2_zero = torch.tensor(metadata["zeros"]["classifier.3.weight"])

    def _dequantise(self, weight: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
        """Dequantise INT8 weight to FP32."""
        return (weight.float() - zero.to(weight.device)) * scale.to(weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly dequantisation."""
        import torch.nn.functional as F

        # Conv1 + BN1 + ReLU + MaxPool
        w1 = self._dequantise(self.conv1_weight, self.conv1_scale, self.conv1_zero)
        x = F.conv2d(x, w1, self.conv1_bias, stride=1, padding=1)
        x = F.batch_norm(
            x, self.bn1_mean, self.bn1_var, self.bn1_weight, self.bn1_bias, training=False
        )
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Conv2 + BN2 + ReLU + MaxPool
        w2 = self._dequantise(self.conv2_weight, self.conv2_scale, self.conv2_zero)
        x = F.conv2d(x, w2, self.conv2_bias, stride=1, padding=1)
        x = F.batch_norm(
            x, self.bn2_mean, self.bn2_var, self.bn2_weight, self.bn2_bias, training=False
        )
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Conv3 + BN3 + ReLU + MaxPool
        w3 = self._dequantise(self.conv3_weight, self.conv3_scale, self.conv3_zero)
        x = F.conv2d(x, w3, self.conv3_bias, stride=1, padding=1)
        x = F.batch_norm(
            x, self.bn3_mean, self.bn3_var, self.bn3_weight, self.bn3_bias, training=False
        )
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Global average pool
        x = x.mean(dim=[2, 3])

        # FC1 + ReLU + Dropout (no dropout in inference)
        w_fc1 = self._dequantise(self.fc1_weight, self.fc1_scale, self.fc1_zero)
        x = F.linear(x, w_fc1, self.fc1_bias)
        x = F.relu(x)

        # FC2
        w_fc2 = self._dequantise(self.fc2_weight, self.fc2_scale, self.fc2_zero)
        x = F.linear(x, w_fc2, self.fc2_bias)

        return x


def main() -> None:
    """Main export function."""
    checkpoint_path = Path("data/experiments/phase4b/tiny_cnn_kd_int8.pt")

    if not checkpoint_path.exists():
        logger.error(f"INT8 checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    output_dir = Path("web_demo/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Calculate storage size (INT8 weights + FP32 biases/scales)
    compressed = checkpoint["model"]
    metadata = checkpoint["metadata"]

    int8_weight_bytes = sum(
        p.numel() for name, p in compressed.items() if name in metadata["scales"]
    )  # 1 byte each
    fp32_param_bytes = sum(
        p.numel() * 4
        for name, p in compressed.items()
        if name not in metadata["scales"]
    )
    total_bytes = int8_weight_bytes + fp32_param_bytes

    logger.info(f"INT8 model storage: {total_bytes / 1024:.1f} KB")
    logger.info(f"  - INT8 weights: {int8_weight_bytes / 1024:.1f} KB")
    logger.info(f"  - FP32 params: {fp32_param_bytes / 1024:.1f} KB")

    # Create and export model
    model = INT8CompactCNN(checkpoint)
    model.eval()

    # Dummy input
    dummy_input = torch.randn(1, 1, 80, 100)

    output_path = output_dir / "model_int8.onnx"
    logger.info(f"Exporting to {output_path}")

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
    logger.info(f"Exported ONNX file: {file_size / 1024:.1f} KB")
    logger.info(
        f"Note: INT8 weights stored in graph, dequantised during forward pass"
    )


if __name__ == "__main__":
    main()
