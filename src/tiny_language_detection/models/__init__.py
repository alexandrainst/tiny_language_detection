"""Model architectures for language detection."""

from .cnn import LanguageDetectionCNN, count_parameters

__all__ = ["LanguageDetectionCNN", "count_parameters"]
