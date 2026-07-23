"""CNN-RNN model for language detection.

This model combines convolutional layers for local feature extraction with recurrent
layers (GRU) for temporal modelling. The architecture follows best practices for
edge-compatible audio classification [Cerna et al., 2023; Jiang et al., 2025].

The CNN extracts time-frequency patterns from the Log Mel-spectrogram input, while
the GRU layer captures longer-range temporal dependencies that may improve language
discrimination, especially for longer utterances.

References:
    Cerna, P. et al. (2023). An IoT-Based Language Recognition System For Indigenous
    Languages Using Integrated CNN And RNN.

    Jiang, X. et al. (2025). M3Net: Efficient Time-Frequency Integration Network With
    Mirror Attention For Audio Classification On Edge. AAAI.

    Ezilarasan, K. et al. (2026). Audio Language Detection Using Deep Learning.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNNRNNLanguageDetector(nn.Module):
    """CNN-RNN architecture for language detection.

    Architecture:
        Input: [batch, n_mels=80, time_steps, 1]

        CNN Blocks (3):
            Conv2d + BatchNorm + ReLU
            Conv2d + BatchNorm + ReLU
            MaxPool(2, 2)

        After CNN: spatial dimensions reduced by 8x

        RNN:
            GRU (1-2 layers, 64-128 hidden units)
            Bidirectional=False (for edge efficiency)

        Classifier:
            Linear(hidden_size, num_languages)

    Attributes:
        cnn: Sequential CNN feature extractor.
        rnn: GRU layer for temporal modelling.
        classifier: Final linear layer for language prediction.

    Example:
        >>> model = CNNRNNLanguageDetector(n_mels=80, hidden_size=64)
        >>> x = torch.randn(8, 80, 50, 1)  # batch=8, time=50
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([8, 2])
    """

    def __init__(
        self,
        n_mels: int = 80,
        channels: list[int] = None,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_languages: int = 2,
        dropout: float = 0.3,
    ) -> None:
        """Initialise the CNN-RNN model.

        Args:
            n_mels: Number of input Mel bins (typically 64-128).
            channels: List of channel counts per CNN block (e.g., [32, 64, 128]).
                     Each block has 2 conv layers with that many channels.
            hidden_size: GRU hidden dimension (64-128 recommended).
            num_layers: Number of GRU layers (1-2 recommended).
            num_languages: Number of output classes (default 2 for DA/EN).
            dropout: Dropout probability (default: 0.3).
        """
        super().__init__()

        if channels is None:
            channels = [32, 64, 128]

        self.n_mels = n_mels
        self.channels = channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_languages = num_languages

        # CNN feature extractor (configurable blocks)
        cnn_blocks = []
        in_channels = 1
        for block_channels in channels:
            # Two conv layers per block
            cnn_blocks.extend([
                nn.Conv2d(in_channels, block_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(block_channels),
                nn.ReLU(),
                nn.Conv2d(block_channels, block_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(block_channels),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            ])
            in_channels = block_channels

        self.cnn = nn.Sequential(*cnn_blocks)

        # Calculate CNN output dimensions
        # After len(channels) max pools of 2x2: n_mels // (2**len(channels))
        freq_dim = n_mels // (2 ** len(channels))
        self.freq_feature_size = channels[-1] * freq_dim  # After flattening CNN output

        # RNN for temporal modelling
        self.rnn = nn.GRU(
            input_size=self.freq_feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,  # Unidirectional for efficiency
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_size, num_languages)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape `[batch, n_mels, time]` or
                `[batch, 1, n_mels, time]`.

        Returns:
            Logits tensor of shape `[batch, num_languages]`.
        """
        # Ensure input is [batch, 1, n_mels, time] for Conv2d
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [batch, 1, n_mels, time]
        elif x.dim() == 4 and x.size(1) != 1:
            # If [batch, n_mels, time, 1], permute to [batch, 1, n_mels, time]
            x = x.permute(0, 3, 1, 2)

        # CNN feature extraction
        x = self.cnn(x)  # [batch, channels[-1], freq_dim, time_dim]

        # Rearrange for RNN: [batch, time, features]
        batch_size = x.size(0)
        x = x.permute(0, 3, 1, 2)  # [batch, time_dim, channels[-1], freq_dim]
        x = x.reshape(batch_size, x.size(1), -1)  # [batch, time_dim, channels[-1]*freq_dim]

        # RNN temporal modelling
        _, hidden = self.rnn(x)  # hidden: [num_layers, batch, hidden_size]

        # Use last layer's last hidden state
        x = hidden[-1]  # [batch, hidden_size]

        # Classification
        logits = self.classifier(x)  # [batch, num_languages]

        return logits

    def count_parameters(self) -> int:
        """Count trainable parameters.

        Returns:
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_cnn_rnn_model(
    n_mels: int = 80,
    channels: list[int] = None,
    hidden_size: int = 64,
    num_layers: int = 1,
    num_languages: int = 2,
    dropout: float = 0.3,
) -> CNNRNNLanguageDetector:
    """Factory function for creating CNN-RNN models.

    Args:
        n_mels: Number of Mel bins for input features.
        channels: List of channel counts per CNN block (e.g., [32, 64, 128]).
        hidden_size: GRU hidden dimension.
        num_layers: Number of GRU layers.
        num_languages: Number of output classes.
        dropout: Dropout probability.

    Returns:
        Instantiated CNNRNNLanguageDetector model.

    Example:
        >>> model = create_cnn_rnn_model(n_mels=80, channels=[32, 64], hidden_size=64)
        >>> print(f"Model has {model.count_parameters():,} parameters")
    """
    return CNNRNNLanguageDetector(
        n_mels=n_mels,
        channels=channels,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_languages=num_languages,
        dropout=dropout,
    )
