"""MFCC feature extraction for audio language detection."""

import torch
import torchaudio.transforms as T

from ..config import Config


def extract_mfcc(
    waveform: torch.Tensor, sample_rate: int, config: Config
) -> torch.Tensor:
    """Extract MFCC features from an audio waveform.

    Args:
        waveform:
            Input audio waveform with shape [channels, time_steps].
        sample_rate:
            Sample rate of the audio in Hz.
        config:
            Configuration object containing MFCC parameters.

    Returns:
        MFCC features with shape [num_mfcc, time_steps].

    Raises:
        ValueError:
            If mfcc_num_coeffs is not in the valid range 13-40.
    """
    num_coeffs = config.mfcc_num_coeffs

    if num_coeffs < 13 or num_coeffs > 40:
        raise ValueError(f"mfcc_num_coeffs must be in range 13-40, got {num_coeffs}")

    hop_length = int(sample_rate * config.mfcc_hop_ms / 1000)
    n_fft = int(sample_rate * config.mfcc_window_ms / 1000)

    mfcc_transform = T.MFCC(
        sample_rate=config.sample_rate,
        n_mfcc=num_coeffs,
        melkwargs={"n_mels": 40, "hop_length": hop_length, "n_fft": n_fft},
    )

    mfccs = mfcc_transform(waveform)

    if mfccs.dim() == 3:
        mfccs = mfccs.squeeze(0)

    return mfccs
