#!/usr/bin/env python3
"""Compress Phase 4b models to different precisions.

Usage:
    # BF16
    uv run src/scripts/phase4b_compression.py \\
        --checkpoint data/experiments/phase4b/tiny_cnn_kd/model_best.pth \\
        --precision bf16 \\
        --output data/experiments/phase4b/tiny_cnn_kd_bf16.pt

    # INT8 (weight-only quantisation)
    uv run src/scripts/phase4b_compression.py \\
        --checkpoint data/experiments/phase4b/tiny_cnn_kd/model_best.pth \\
        --precision int8 \\
        --output data/experiments/phase4b/tiny_cnn_kd_int8.pt

    # INT4 (weight-only quantisation)
    uv run src/scripts/phase4b_compression.py \\
        --checkpoint data/experiments/phase4b/tiny_cnn_kd/model_best.pth \\
        --precision int4 \\
        --output data/experiments/phase4b/tiny_cnn_kd_int4.pt
"""

import argparse
import json
import logging
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def quantise_to_int8(state_dict: dict) -> tuple[dict, dict]:
    """Quantise weights to 8-bit integers.

    Args:
        state_dict: Original FP32 state dict.

    Returns:
        Tuple of (quantised state dict, metadata for dequantisation).
    """
    quantised = {}
    metadata = {"scales": {}, "zeros": {}}

    for name, param in state_dict.items():
        if param.dtype == torch.float32 and param.dim() >= 2 and param.numel() > 100:
            # Per-tensor quantisation
            min_val = torch.min(param)
            max_val = torch.max(param)
            scale = (max_val - min_val) / 255.0
            zero_point = (-min_val / scale).round().clamp(0, 255).to(torch.int32)
            q = ((param / scale) + zero_point).round().clamp(0, 255).to(torch.uint8)
            quantised[name] = q
            metadata["scales"][name] = float(scale.item())
            metadata["zeros"][name] = int(zero_point.item())
        else:
            # Keep small tensors (biases, etc.) in FP32
            quantised[name] = param.clone()

    return quantised, metadata


def quantise_to_int4(state_dict: dict) -> tuple[dict, dict]:
    """Quantise weights to 4-bit integers (16 levels).

    Args:
        state_dict: Original FP32 state dict.

    Returns:
        Tuple of (quantised state dict, metadata for dequantisation).
    """
    quantised = {}
    metadata = {"scales": {}, "zeros": {}}

    for name, param in state_dict.items():
        if param.dim() >= 2 and param.numel() > 100:
            # Per-tensor quantisation to 4-bit (16 levels)
            min_val = torch.min(param)
            max_val = torch.max(param)
            scale = (max_val - min_val) / 15.0
            zero_point = (-min_val / scale).round().clamp(0, 15).to(torch.int8)
            q = (param / scale + zero_point).round().clamp(0, 15).to(torch.int8)
            quantised[name] = q
            metadata["scales"][name] = scale.item()
            metadata["zeros"][name] = int(zero_point.item())
        else:
            # Keep small tensors in FP32
            quantised[name] = param.clone()

    return quantised, metadata


def dequantise_int8(
    quantised: dict,
    metadata: dict,
    device: torch.device,
) -> dict:
    """Dequantise INT8 weights to FP16 for inference.

    Args:
        quantised: Quantised state dict.
        metadata: Quantisation metadata (scales, zeros).
        device: Device to load weights on.

    Returns:
        Dequantised state dict in BF16.
    """
    state = {}
    for name, param in quantised.items():
        if name in metadata["scales"]:
            scale = torch.tensor(metadata["scales"][name], device=device)
            zero = torch.tensor(metadata["zeros"][name], device=device)
            state[name] = (param.to(device).float() - zero) * scale
        else:
            state[name] = param.to(device)
    return state


def dequantise_int4(
    quantised: dict,
    metadata: dict,
    device: torch.device,
) -> dict:
    """Dequantise INT4 weights to FP16 for inference.

    Args:
        quantised: Quantised state dict.
        metadata: Quantisation metadata (scales, zeros).
        device: Device to load weights on.

    Returns:
        Dequantised state dict in BF16.
    """
    state = {}
    for name, param in quantised.items():
        if name in metadata["scales"]:
            scale = torch.tensor(metadata["scales"][name], device=device)
            zero = torch.tensor(metadata["zeros"][name], device=device)
            state[name] = (param.to(device).float() - zero) * scale
        else:
            state[name] = param.to(device)
    return state


def compress_model(
    checkpoint_path: Path,
    output_path: Path,
    precision: str,
) -> dict:
    """Compress a model to specified precision.

    Args:
        checkpoint_path: Path to original FP32 checkpoint.
        output_path: Path to save compressed model.
        precision: Target precision ('bf16', 'int8', 'int4').

    Returns:
        Dictionary with compression metrics.
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config = checkpoint.get("config", {})
    else:
        state_dict = checkpoint
        config = {}

    # Count original parameters
    original_params = sum(p.numel() for p in state_dict.values())
    original_size = original_params * 4  # FP32 = 4 bytes

    logger.info(f"Original model: {original_params:,} params, {original_size / 1024:.1f} KB")

    # Compress based on precision
    if precision == "bf16":
        # BF16 conversion
        compressed = {
            name: param.to(torch.bfloat16) if param.dtype == torch.float32 else param
            for name, param in state_dict.items()
        }
        metadata = {}

        # Calculate size (BF16 = 2 bytes)
        bf16_params = sum(
            p.numel() for p in compressed.values() if p.dtype == torch.bfloat16
        )
        fp32_params = sum(
            p.numel() for p in compressed.values() if p.dtype == torch.float32
        )
        compressed_size = bf16_params * 2 + fp32_params * 4

    elif precision == "int8":
        # INT8 quantisation
        compressed, metadata = quantise_to_int8(state_dict)

        # Calculate size (INT8 = 1 byte, FP32 kept = 4 bytes)
        int8_params = sum(
            p.numel() for p in compressed.values() if p.dtype == torch.uint8
        )
        fp32_params = sum(
            p.numel() for p in compressed.values() if p.dtype == torch.float32
        )
        compressed_size = int8_params * 1 + fp32_params * 4

    elif precision == "int4":
        # INT4 quantisation
        compressed, metadata = quantise_to_int4(state_dict)

        # Calculate size (INT4 = 0.5 bytes, FP32 kept = 4 bytes)
        int4_params = sum(
            p.numel() for p in compressed.values() if p.dtype == torch.int8
        )
        fp32_params = sum(
            p.numel() for p in compressed.values() if p.dtype == torch.float32
        )
        compressed_size = int4_params * 0.5 + fp32_params * 4

    else:
        raise ValueError(f"Unknown precision: {precision}")

    logger.info(f"Compressed model: {compressed_size / 1024:.1f} KB")
    logger.info(f"Compression ratio: {original_size / compressed_size:.2f}x")

    # Add precision to config
    config["precision"] = precision

    # Save compressed model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": compressed, "metadata": metadata, "config": config},
        output_path,
    )
    logger.info(f"Saved to {output_path}")

    return {
        "original_size_kb": original_size / 1024,
        "compressed_size_kb": compressed_size / 1024,
        "compression_ratio": original_size / compressed_size,
        "precision": precision,
        "original_params": original_params,
    }


def main() -> None:
    """Main compression function."""
    parser = argparse.ArgumentParser(description="Compress Phase 4b models")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to FP32 checkpoint",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["bf16", "int8", "int4"],
        required=True,
        help="Target precision",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for compressed model",
    )
    args = parser.parse_args()

    metrics = compress_model(
        Path(args.checkpoint),
        Path(args.output),
        args.precision,
    )

    # Save metrics
    metrics_path = Path(args.output).with_suffix(".json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
