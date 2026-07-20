"""Lightweight CNN for audio language detection from MFCC features."""

import torch
import torch.nn as nn


class LanguageDetectionCNN(nn.Module):
    """Lightweight CNN for binary/multiclass language detection.

    Accepts MFCC input tensors and outputs class logits. Designed to have
    50k-100k trainable parameters while maintaining good classification
    performance.

    Architecture:
        - 3 convolutional blocks (Conv2d + BatchNorm + ReLU + MaxPool)
        - Global average pooling
        - Dense classification head

    Input shape:
        [batch_size, num_mfcc, time_steps, 1]

    Output shape:
        [batch_size, num_languages]

    Note:
        MFCC+CNN pipelines validated for speaker-independent speech
        classification (Zhu et al., 2025).
    """

    def __init__(
        self, num_mfcc: int = 40, time_steps: int = 50, num_languages: int = 2
    ) -> None:
        """Initialise the CNN.

        Args:
            num_mfcc:
                Number of MFCC features (typically 40).
            time_steps:
                Number of time steps in the MFCC sequence.
            num_languages:
                Number of language classes to predict.
        """
        super().__init__()

        self.num_mfcc = num_mfcc
        self.time_steps = time_steps
        self.num_languages = num_languages

        # Convolutional blocks designed for ~50k-100k parameters
        # Block 1: 1 -> 32 channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)

        # Block 3: 64 -> 64 channels
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global average pooling replaces flattening + large FC layer
        # This keeps parameter count low while preserving spatial information
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.classifier = nn.Linear(64, num_languages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x:
                Input MFCC tensor of shape
                [batch_size, num_mfcc, time_steps, 1].

        Returns:
            Logits tensor of shape [batch_size, num_languages].
        """
        # Ensure input is in NCHW format
        # If input is [batch, mfcc, time, 1], transpose to [batch, 1, mfcc, time]
        if x.dim() == 4 and x.shape[3] == 1:
            x = x.permute(0, 3, 1, 2)

        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Global average pooling: [batch, 64, h, w] -> [batch, 64, 1, 1]
        x = self.global_pool(x)

        # Flatten: [batch, 64, 1, 1] -> [batch, 64]
        x = x.view(x.size(0), -1)

        # Classification head: [batch, 64] -> [batch, num_languages]
        logits = self.classifier(x)

        return logits


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a PyTorch model.

    Args:
        model:
            PyTorch model to count parameters for.

    Returns:
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test with dummy MFCC batch
    batch_size = 4
    num_mfcc = 40
    time_steps = 50
    num_languages = 2

    model = LanguageDetectionCNN(
        num_mfcc=num_mfcc, time_steps=time_steps, num_languages=num_languages
    )

    # Create dummy input
    dummy_input = torch.randn(batch_size, num_mfcc, time_steps, 1)

    # Forward pass
    logits = model(dummy_input)

    # Count parameters
    param_count = count_parameters(model)

    print(f"Model parameter count: {param_count:,}")
    print("Target range: 50,000 - 100,000")
    print(f"Within target: {50000 <= param_count <= 100000}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {logits.shape}")
    expected_shape = f"[{batch_size}, {num_languages}]"
    print(f"Expected output shape: [batch_size, num_languages] = {expected_shape}")
