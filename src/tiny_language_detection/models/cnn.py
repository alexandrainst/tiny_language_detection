"""Compact CNN models for low-RAM deployment.

Three model sizes for different RAM budgets:
- Tiny: ~50k params, ~200 KB RAM, target 80-85% accuracy
- Small: ~100k params, ~400 KB RAM, target 85-88% accuracy
- Medium: ~200k params, ~800 KB RAM, target 88-90% accuracy

All use CNN + global pooling (no RNN/GRU) for simplicity and efficiency.
"""

import torch
import torch.nn as nn


class CompactCNNLanguageDetector(nn.Module):
    """Compact CNN for language detection on resource-constrained devices.

    Scalable architecture with configurable depth and width.

    Architecture:
        Input: [batch, 1, n_mels=80, time_steps]

        CNN Blocks (configurable):
            Conv2d + BatchNorm + ReLU + MaxPool (× N blocks)
            GlobalAvgPool → [batch, final_channels]

        Classifier:
            Linear(final_channels, hidden) + ReLU + Dropout
            Linear(hidden, num_languages)

    Example:
        >>> model = CompactCNNLanguageDetector(
        ...     n_mels=80,
        ...     channels=[32, 64, 128],  # 3 blocks
        ...     hidden_size=64,
        ... )
        >>> x = torch.randn(4, 1, 80, 50)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([4, 2])
    """

    def __init__(
        self,
        n_mels: int = 80,
        channels: list[int] | None = None,
        hidden_size: int = 64,
        num_languages: int = 2,
        dropout: float = 0.3,
    ) -> None:
        """Initialise the compact CNN model.

        Args:
            n_mels: Number of input Mel bins (typically 64-80).
            channels: List of channel counts per block (e.g., [32, 64, 128]).
            hidden_size: Hidden layer size in classifier.
            num_languages: Number of output classes.
            dropout: Dropout probability for regularisation.
        """
        super().__init__()

        if channels is None:
            channels = [32, 64, 128]

        self.n_mels = n_mels
        self.num_languages = num_languages
        self.channels = channels

        # Build CNN blocks
        blocks = []
        in_channels = 1
        for i, out_channels in enumerate(channels):
            blocks.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),  # Halves frequency dimension
                ]
            )
            in_channels = out_channels

        self.features = nn.Sequential(*blocks)

        # Calculate flattened size after CNN + pooling
        # Each pool halves: 80 → 40 → 20 → 10 → 5
        freq_after_pool = n_mels // (2 ** len(channels))
        self.flattened_size = channels[-1] * max(1, freq_after_pool)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_languages),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape `[batch, 1, n_mels, time]` or
                `[batch, n_mels, time]` (channel dimension added automatically).

        Returns:
            Logits tensor of shape `[batch, num_languages]`.
        """
        # Ensure input is [batch, 1, n_mels, time] for Conv2d
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Feature extraction
        x = self.features(x)  # [batch, channels[-1], freq, time]

        # Global average pooling over time dimension
        x = x.mean(dim=3)  # [batch, channels[-1], freq]

        # Flatten
        x = x.view(x.size(0), -1)  # [batch, channels[-1] * freq]

        # Classification
        logits = self.classifier(x)  # [batch, num_languages]

        return logits

    def count_parameters(self) -> int:
        """Count trainable parameters.

        Returns:
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_tiny_cnn(num_languages: int = 2) -> CompactCNNLanguageDetector:
    """Create a tiny model (~50k params, ~200 KB RAM).

    Target: 80-85% accuracy for very constrained devices.

    Args:
        num_languages: Number of output classes.

    Returns:
        Tiny CompactCNNLanguageDetector.
    """
    return CompactCNNLanguageDetector(
        n_mels=80,
        channels=[16, 32, 64],  # 3 blocks, 64 final channels
        hidden_size=32,  # Small hidden layer
        num_languages=num_languages,
        dropout=0.3,
    )


def create_small_cnn(num_languages: int = 2) -> CompactCNNLanguageDetector:
    """Create a small model (~100k params, ~400 KB RAM).

    Target: 85-88% accuracy for moderate constraints.

    Args:
        num_languages: Number of output classes.

    Returns:
        Small CompactCNNLanguageDetector.
    """
    return CompactCNNLanguageDetector(
        n_mels=80,
        channels=[32, 64, 128],  # 3 blocks, 128 final channels
        hidden_size=64,  # Medium hidden layer
        num_languages=num_languages,
        dropout=0.3,
    )


def create_medium_cnn(num_languages: int = 2) -> CompactCNNLanguageDetector:
    """Create a medium model (~150k params, ~600 KB RAM).

    Target: 88-90% accuracy, competing with Phase 2.

    Args:
        num_languages: Number of output classes.

    Returns:
        Medium CompactCNNLanguageDetector.
    """
    return CompactCNNLanguageDetector(
        n_mels=80,
        channels=[32, 64, 128],  # 3 blocks (same as small)
        hidden_size=256,  # Larger hidden layer for more capacity
        num_languages=num_languages,
        dropout=0.4,
    )


# Backward compatibility alias
TinyCNNLanguageDetector = create_tiny_cnn
