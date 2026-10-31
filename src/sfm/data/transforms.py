"""Data augmentations, spectral normalization, and contiguous patch masking."""

import torch


class PatchMasker:
    """Generates random boolean masks for Masked Spectral Autoencoding."""

    def __init__(self, num_patches: int, mask_ratio: float = 0.75) -> None:
        """Initialize PatchMasker.

        Args:
            num_patches: Total number of non-overlapping spectral patches.
            mask_ratio: Fraction of patches to mask (default: 0.75).
        """
        self.num_patches = int(num_patches)
        self.mask_ratio = float(mask_ratio)
        self.num_masked = int(self.num_patches * self.mask_ratio)

    def __call__(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """Generate boolean mask tensor of shape (batch_size, num_patches).

        Args:
            batch_size: Number of spectra in batch.
            device: Target device for tensor.

        Returns:
            Boolean tensor where True indicates a masked patch position.
        """
        # Generate random noise and find permutations
        noise = torch.rand(batch_size, self.num_patches, device=device)
        perm = torch.argsort(noise, dim=-1)

        mask = torch.zeros(batch_size, self.num_patches, dtype=torch.bool, device=device)
        # Select first num_masked indices from the random permutation
        masked_indices = perm[:, : self.num_masked]
        mask.scatter_(dim=-1, index=masked_indices, value=True)
        return mask


class SpectralNormalizer:
    """Normalizes continuous spectral intensity across the wavenumber dimension."""

    def __init__(self, method: str = "minmax") -> None:
        """Initialize SpectralNormalizer.

        Args:
            method: Normalization strategy ('minmax' or 'standard').
        """
        self.method = method.lower()

    def __call__(self, spectra: torch.Tensor) -> torch.Tensor:
        """Apply normalization along the last dimension.

        Args:
            spectra: Tensor of shape (..., channels).

        Returns:
            Normalized tensor of identical shape.
        """
        if self.method == "minmax":
            mins = spectra.amin(dim=-1, keepdim=True)
            maxs = spectra.amax(dim=-1, keepdim=True)
            ranges = maxs - mins
            ranges = torch.where(ranges == 0.0, torch.ones_like(ranges), ranges)
            return (spectra - mins) / ranges

        elif self.method == "standard":
            means = spectra.mean(dim=-1, keepdim=True)
            stds = spectra.std(dim=-1, keepdim=True)
            stds = torch.where(stds == 0.0, torch.ones_like(stds), stds)
            return (spectra - means) / stds

        return spectra


def add_gaussian_noise(spectra: torch.Tensor, snr: float) -> torch.Tensor:
    """Inject additive Gaussian white noise matching a specified Signal-to-Noise Ratio.

    Args:
        spectra: Spectral tensor of shape (..., channels).
        snr: Signal-to-noise ratio (e.g. 500 to 3000).

    Returns:
        Perturbed spectra tensor.
    """
    signal_std = spectra.std(dim=-1, keepdim=True)
    # If spectrum has zero variance (constant baseline), fallback to mean absolute amplitude
    zero_var = (signal_std == 0.0)
    if zero_var.any():
        fallback = spectra.abs().mean(dim=-1, keepdim=True)
        fallback = torch.where(fallback == 0.0, torch.ones_like(fallback), fallback)
        signal_std = torch.where(zero_var, fallback, signal_std)

    noise_std = signal_std / max(float(snr), 1e-4)
    noise = torch.randn_like(spectra) * noise_std
    return spectra + noise
