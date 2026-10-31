"""Unit tests for dataset loaders in sfm.data.dataset."""

import os
import torch
from sfm.data.dataset import LabeledSpectraDataset, SpectraDataset

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
MINI_SPECTRA_PATH = os.path.join(FIXTURES_DIR, "mini_spectra.npy")
MINI_LABELS_PATH = os.path.join(FIXTURES_DIR, "mini_labels.npz")


def test_spectral_dataset_length():
    """SpectraDataset must correctly report length corresponding to number of samples."""
    dataset = SpectraDataset(MINI_SPECTRA_PATH, num_samples=100, num_channels=200)
    assert len(dataset) == 100


def test_spectral_dataset_getitem():
    """SpectraDataset items must be float32 1D tensors with expected channel length."""
    dataset = SpectraDataset(MINI_SPECTRA_PATH, num_samples=100, num_channels=200)
    item = dataset[0]
    assert isinstance(item, torch.Tensor)
    assert item.shape == (200,)
    assert item.dtype == torch.float32


def test_spectral_dataset_getitem_range():
    """SpectraDataset values must be finite without NaN or Inf values."""
    dataset = SpectraDataset(MINI_SPECTRA_PATH, num_samples=100, num_channels=200)
    item = dataset[0]
    assert torch.isfinite(item).all()


def test_labeled_dataset_length():
    """LabeledSpectraDataset must correctly report labeled sample count."""
    dataset = LabeledSpectraDataset(MINI_LABELS_PATH)
    assert len(dataset) == 20


def test_labeled_dataset_getitem():
    """LabeledSpectraDataset must return spectrum, binary labels, and concentrations tuple."""
    dataset = LabeledSpectraDataset(MINI_LABELS_PATH)
    spectrum, labels, concentrations = dataset[0]
    assert isinstance(spectrum, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert isinstance(concentrations, torch.Tensor)
    assert spectrum.shape == (200,)
    assert labels.shape == (5,)
    assert concentrations.shape == (5,)
    assert torch.isfinite(spectrum).all()

