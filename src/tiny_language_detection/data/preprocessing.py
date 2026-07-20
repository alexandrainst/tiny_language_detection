"""Audio preprocessing utilities.

Functions for loading and preprocessing audio files for language detection.
"""

from pathlib import Path

import torchaudio
from torch import Tensor


def load_and_preprocess(audio_path: str | Path, target_sr: int = 16000) -> Tensor:
    """Load and preprocess an audio file.

    Loads audio using torchaudio, resamples to target sample rate if needed,
    and converts to mono by averaging across channels.

    Args:
        audio_path:
            Path to the audio file.
        target_sr (optional):
            Target sample rate. Defaults to 16000.

    Returns:
        Audio tensor of shape (1, num_samples) at target sample rate.

    Raises:
        FileNotFoundError:
            If the audio file does not exist.
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert to mono if stereo (average across channels)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sample_rate, new_freq=target_sr
        )

    return waveform


def get_audio_duration(audio_path: str | Path, target_sr: int = 16000) -> float:
    """Get the duration of an audio file in seconds.

    Args:
        audio_path:
            Path to the audio file.
        target_sr (optional):
            Sample rate to use for duration calculation. Defaults to 16000.

    Returns:
        Duration in seconds.

    Raises:
        FileNotFoundError:
            If the audio file does not exist.
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load audio to get duration (without resampling)
    waveform, sample_rate = torchaudio.load(audio_path)
    num_samples = waveform.shape[-1]
    duration = num_samples / sample_rate

    return duration


def assign_duration_group(duration: float) -> str:
    """Assign a duration group based on audio length.

    Groups:
        - "0-2s": 0 to 2 seconds (exclusive)
        - "2-4s": 2 to 4 seconds (exclusive)
        - "4-6s": 4 to 6 seconds (exclusive)
        - "6+s": 6 seconds and above

    Args:
        duration:
            Duration in seconds.

    Returns:
        Duration group string.
    """
    if duration < 2:
        return "0-2s"
    elif duration < 4:
        return "2-4s"
    elif duration < 6:
        return "4-6s"
    else:
        return "6+s"
