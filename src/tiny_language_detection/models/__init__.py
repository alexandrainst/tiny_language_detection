"""Model architectures for language detection."""

from .cnn import LanguageDetectionCNN, count_parameters
from .cnn_rnn import CNNRNNLanguageDetector, create_cnn_rnn_model

__all__ = [
    "LanguageDetectionCNN",
    "count_parameters",
    "CNNRNNLanguageDetector",
    "create_cnn_rnn_model",
]
