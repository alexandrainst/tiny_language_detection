"""Test script to verify the CNN model implementation.

This script validates that the LanguageDetectionCNN model:
- Accepts MFCC input tensors correctly
- Produces properly shaped logits
- Has parameter count within the 50k-100k target range
"""

import torch

from tiny_language_detection.models.cnn import LanguageDetectionCNN, count_parameters


def test_forward_pass() -> None:
    """Verify forward pass works on dummy MFCC batch."""
    model = LanguageDetectionCNN(num_languages=2, num_mfcc=40)
    model.eval()

    batch_size = 8
    time_steps = 64
    mfcc_input = torch.randn(batch_size, 40, time_steps, 1)

    with torch.no_grad():
        logits = model(mfcc_input)

    assert logits.shape == (batch_size, 2), (
        f"Expected shape ({batch_size}, 2), got {logits.shape}"
    )
    print(f"✓ Forward pass: input {mfcc_input.shape} -> output {logits.shape}")


def test_multiclass_output() -> None:
    """Verify model works for multiclass classification."""
    num_languages = 5
    model = LanguageDetectionCNN(num_languages=num_languages, num_mfcc=40)
    model.eval()

    batch_size = 4
    time_steps = 32
    mfcc_input = torch.randn(batch_size, 40, time_steps, 1)

    with torch.no_grad():
        logits = model(mfcc_input)

    assert logits.shape == (batch_size, num_languages), (
        f"Expected shape ({batch_size}, {num_languages}), got {logits.shape}"
    )
    print(
        f"✓ Multiclass: {num_languages} languages, "
        f"input {mfcc_input.shape} -> output {logits.shape}"
    )


def test_parameter_count() -> None:
    """Verify parameter count is within 50k-100k target."""
    model = LanguageDetectionCNN(num_languages=2, num_mfcc=40)
    param_count = count_parameters(model)

    min_params = 50_000
    max_params = 100_000

    assert min_params <= param_count <= max_params, (
        f"Parameter count {param_count:,} outside target range "
        f"{min_params:,}-{max_params:,}"
    )
    print(f"✓ Parameter count: {param_count:,} (target: 50k-100k)")


def test_variable_time_steps() -> None:
    """Verify model handles variable time steps."""
    model = LanguageDetectionCNN(num_languages=2, num_mfcc=40)
    model.eval()

    for time_steps in [32, 64, 128]:
        mfcc_input = torch.randn(2, 40, time_steps, 1)

        with torch.no_grad():
            logits = model(mfcc_input)

        assert logits.shape == (2, 2), (
            f"Failed for time_steps={time_steps}: got {logits.shape}"
        )
        print(f"✓ Variable time steps ({time_steps}): output {logits.shape}")


def main() -> None:
    """Run all model tests."""
    print("Testing LanguageDetectionCNN model...\n")

    test_forward_pass()
    test_multiclass_output()
    test_parameter_count()
    test_variable_time_steps()

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    main()
