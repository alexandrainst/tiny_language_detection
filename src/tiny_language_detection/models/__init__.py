"""Model architectures for language detection."""

from .cnn import (
    CompactCNNLanguageDetector,
    create_medium_cnn,
    create_small_cnn,
    create_tiny_cnn,
)
from .cnn_rnn import CNNRNNLanguageDetector, create_cnn_rnn_model

__all__ = [
    "CompactCNNLanguageDetector",
    "create_medium_cnn",
    "create_small_cnn",
    "create_tiny_cnn",
    "CNNRNNLanguageDetector",
    "create_cnn_rnn_model",
]
