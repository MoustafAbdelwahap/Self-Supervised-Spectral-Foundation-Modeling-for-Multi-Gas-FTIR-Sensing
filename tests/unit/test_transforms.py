"""Unit tests for transforms and patch masking in sfm.data.transforms and sfm.models.mae."""

import torch
from sfm.data.transforms import PatchMasker, SpectralNormalizer, add_gaussian_noise
from sfm.models.mae import PatchEmbedding


def test_patch_mask_shape():
    """PatchMasker must output boolean tensor with shape (batch_size, num_patches)."""
    masker = PatchMasker(num_patches=20, mask_ratio=0.75)
    mask = masker(batch_size=4)
    assert mask.shape == (4, 20)
    assert mask.dtype == torch.bool


def test_patch_mask_ratio():
    """PatchMasker must mask approximately the specified ratio of patches."""
    num_patches = 20
    mask_ratio = 0.75
    masker = PatchMasker(num_patches=num_patches, mask_ratio=mask_ratio)
    mask = masker(batch_size=10)
    expected_masked_count = int(num_patches * mask_ratio)
    # Check each batch item masks exactly expected_masked_count patches
    for row in mask:
        assert row.sum().item() == expected_masked_count


def test_patch_mask_reproducible():
    """Setting random seed should produce identical masks."""
    torch.manual_seed(42)
    masker = PatchMasker(num_patches=20, mask_ratio=0.75)
    mask1 = masker(batch_size=2)

    torch.manual_seed(42)
    mask2 = masker(batch_size=2)
    torch.testing.assert_close(mask1, mask2)


def test_patch_embed_shape():
    """PatchEmbedding must project continuous 1D spectra into non-overlapping patch tokens."""
    num_channels = 200
    patch_size = 10
    embed_dim = 256
    batch_size = 2

    patch_embed = PatchEmbedding(num_channels=num_channels, patch_size=patch_size, embed_dim=embed_dim)
    x = torch.randn(batch_size, num_channels)
    tokens = patch_embed(x)

    num_patches = num_channels // patch_size
    assert tokens.shape == (batch_size, num_patches, embed_dim)


def test_spectral_normalizer():
    """SpectralNormalizer must normalize spectra to [0, 1] range."""
    normalizer = SpectralNormalizer(method="minmax")
    x = torch.tensor([[10.0, 20.0, 30.0], [5.0, 15.0, 25.0]])
    normed = normalizer(x)
    assert torch.isclose(normed[0].min(), torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(normed[0].max(), torch.tensor(1.0), atol=1e-5)


def test_add_gaussian_noise():
    """add_gaussian_noise must add perturbation according to specified SNR."""
    x = torch.linspace(0.1, 1.0, 50).repeat(2, 1)
    noisy = add_gaussian_noise(x, snr=100.0)
    assert noisy.shape == x.shape
    assert not torch.equal(x, noisy)
