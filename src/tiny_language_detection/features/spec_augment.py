"""SpecAugment: Data augmentation for spectrograms."""

import torch


class SpecAugment:
    """Apply SpecAugment to log-mel spectrograms.

    Args:
        time_mask_param: Maximum number of consecutive time steps to mask.
        freq_mask_param: Maximum number of consecutive frequency channels to mask.
        time_masks: Number of time masks to apply.
        freq_masks: Number of frequency masks to apply.
        inplace: Whether to modify the input tensor in place.

    Example:
        >>> augment = SpecAugment(time_mask_param=5, freq_mask_param=3)
        >>> spec = torch.randn(1, 80, 50)  # (batch, freq, time)
        >>> augmented = augment(spec)
    """

    def __init__(
        self,
        time_mask_param: int = 10,
        freq_mask_param: int = 5,
        time_masks: int = 1,
        freq_masks: int = 1,
        inplace: bool = False,
    ) -> None:
        """Initialise SpecAugment."""
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.time_masks = time_masks
        self.freq_masks = freq_masks
        self.inplace = inplace

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        if spec.dim() == 4:
            return self._apply(spec, channel_dim=1)
        else:
            return self._apply(spec, channel_dim=None)

    def _apply(
        self, spec: torch.Tensor, channel_dim: int | None = None
    ) -> torch.Tensor:
        """Apply masking to spectrogram."""
        # Skip if no masking configured
        if self.time_mask_param <= 0 and self.freq_mask_param <= 0:
            return spec

        if not self.inplace:
            spec = spec.clone()

        if channel_dim is not None:
            batch_size, _, n_freq, n_time = spec.shape
        else:
            batch_size, n_freq, n_time = spec.shape

        # Apply frequency masks
        for _ in range(self.freq_masks):
            if self.freq_mask_param > 0:
                mask_length = torch.randint(1, self.freq_mask_param + 1, (1,)).item()
                if mask_length > 0 and mask_length < n_freq:
                    freq_start = torch.randint(0, n_freq - mask_length + 1, (1,)).item()
                    if channel_dim is not None:
                        spec[:, :, freq_start : freq_start + mask_length, :] = 0
                    else:
                        spec[:, freq_start : freq_start + mask_length, :] = 0

        # Apply time masks
        for _ in range(self.time_masks):
            if self.time_mask_param > 0:
                mask_length = torch.randint(1, self.time_mask_param + 1, (1,)).item()
                if mask_length > 0 and mask_length < n_time:
                    time_start = torch.randint(0, n_time - mask_length + 1, (1,)).item()
                    if channel_dim is not None:
                        spec[:, :, :, time_start : time_start + mask_length] = 0
                    else:
                        spec[:, :, time_start : time_start + mask_length] = 0

        return spec
