"""PyTorch Dataset implementations and DataLoader factories for FTIR spectra."""

import os
from typing import Dict, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split


class SpectraDataset(Dataset):
    """Memory-mapped PyTorch Dataset for large-scale unlabeled spectral pre-training.

    Supports lazy memory-map initialization per DataLoader worker to ensure safe
    multiprocessing and avoid out-of-memory (OOM) errors.
    """

    def __init__(self, memmap_path: str, num_samples: int, num_channels: int) -> None:
        """Initialize SpectraDataset metadata without eagerly mapping into memory.

        Args:
            memmap_path: Path to the .mmap or .npy spectra file.
            num_samples: Total number of spectral rows.
            num_channels: Total number of spectral wavenumber channels.
        """
        super().__init__()
        self.memmap_path = memmap_path
        self.num_samples = int(num_samples)
        self.num_channels = int(num_channels)
        self._mmap: Optional[np.ndarray] = None

    def _ensure_mapped(self) -> np.ndarray:
        if self._mmap is None:
            if self.memmap_path.endswith(".npy"):
                self._mmap = np.load(self.memmap_path, mmap_mode="r")
            else:
                self._mmap = np.memmap(
                    self.memmap_path,
                    dtype="float32",
                    mode="r",
                    shape=(self.num_samples, self.num_channels),
                )
        return self._mmap

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        mmap = self._ensure_mapped()
        spectrum = mmap[idx].copy()
        return torch.from_numpy(spectrum).float()


class LabeledSpectraDataset(Dataset):
    """Eager PyTorch Dataset for downstream multi-gas classification and regression.

    Concentrations are normalized per-gas to [0, 1] using max-scaling so that
    the regression head trains stably via MSE, without gradients dominated by
    gases with large ppm ranges (e.g. H2O at ~25000 ppm vs. N2O at ~800 ppm).

    The per-gas scale factors are stored in ``self.conc_scales`` so callers can
    de-normalize predictions back to raw ppm units for reporting and metrics.
    """

    def __init__(self, npz_path: str, normalize_concentrations: bool = True) -> None:
        """Load labeled spectral dataset into memory.

        Args:
            npz_path: Path to the .npz archive containing spectra, gas_labels, and concentrations.
            normalize_concentrations: If True (default), scale each gas concentration column to
                [0, 1] by its per-gas maximum, improving regression training stability.
        """
        super().__init__()
        self.npz_path = npz_path
        with np.load(npz_path, allow_pickle=True) as data:
            self.spectra = np.asarray(data["spectra"], dtype=np.float32)
            self.gas_labels = np.asarray(data["gas_labels"], dtype=np.float32)
            raw_conc = np.asarray(data["concentrations"], dtype=np.float32)

        # Per-gas max normalization for regression head stability
        self.normalize_concentrations = normalize_concentrations
        if normalize_concentrations:
            self.conc_scales = raw_conc.max(axis=0).clip(min=1.0).astype(np.float32)
            self.concentrations = (raw_conc / self.conc_scales).astype(np.float32)
        else:
            self.conc_scales = np.ones(raw_conc.shape[1], dtype=np.float32)
            self.concentrations = raw_conc

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.spectra[idx].copy()).float(),
            torch.from_numpy(self.gas_labels[idx].copy()).float(),
            torch.from_numpy(self.concentrations[idx].copy()).float(),
        )


def get_conc_scales(data_dir: str) -> np.ndarray:
    """Retrieve per-gas concentration maximum scale factors for unit denormalization.

    Args:
        data_dir: Directory containing processed labeled dataset.

    Returns:
        1D numpy array of per-gas max concentration scale factors in ppm.
    """
    for fname in ("labeled_spectra.npz", "mini_labels.npz"):
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            with np.load(path, allow_pickle=True) as data:
                raw = np.asarray(data["concentrations"], dtype=np.float32)
                return raw.max(axis=0).clip(min=1.0).astype(np.float32)
    return np.ones(5, dtype=np.float32)


def create_dataloaders(
    data_dir: str,
    batch_size: int = 256,
    num_workers: int = 2,
    label_fraction: float = 1.0,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """Create DataLoaders for pre-training, downstream training, validation, and testing.

    Args:
        data_dir: Directory containing processed data files (mmap, npz, npy).
        batch_size: Mini-batch size.
        num_workers: Worker thread count for DataLoaders.
        label_fraction: Fraction of downstream training data to utilize (e.g. 0.1 for 10%).
        seed: Random split seed.

    Returns:
        Dictionary mapping 'pretrain', 'train', 'val', 'test' to DataLoader instances.
    """
    pin_mem = torch.cuda.is_available()

    # 1. Pretrain DataLoader
    mmap_path = os.path.join(data_dir, "pretrain_spectra.mmap")
    wn_path = os.path.join(data_dir, "wavenumbers.npy")
    if not os.path.exists(mmap_path):
        # Fallback to fixtures or raw numpy if mmap not generated yet
        mmap_path = os.path.join(data_dir, "mini_spectra.npy")

    num_channels = len(np.load(wn_path)) if os.path.exists(wn_path) else 200
    if os.path.exists(mmap_path):
        if mmap_path.endswith(".npy"):
            num_samples = len(np.load(mmap_path, mmap_mode="r"))
        else:
            file_bytes = os.path.getsize(mmap_path)
            num_samples = file_bytes // (num_channels * 4)
        pretrain_ds = SpectraDataset(mmap_path, num_samples=num_samples, num_channels=num_channels)
        pretrain_loader = DataLoader(
            pretrain_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=(len(pretrain_ds) > batch_size),
        )
    else:
        pretrain_loader = None

    # 2. Labeled DataLoaders
    labeled_path = os.path.join(data_dir, "labeled_spectra.npz")
    if not os.path.exists(labeled_path):
        labeled_path = os.path.join(data_dir, "mini_labels.npz")

    loaders: Dict[str, DataLoader] = {"pretrain": pretrain_loader}

    if os.path.exists(labeled_path):
        full_labeled_ds = LabeledSpectraDataset(labeled_path)
        indices = list(range(len(full_labeled_ds)))

        # Split 10% / 20% / 70% or proportional for small fixtures
        n_samples = len(indices)
        if n_samples >= 10:
            train_idx, test_val_idx = train_test_split(indices, test_size=0.7, random_state=seed)
            val_idx, test_idx = train_test_split(test_val_idx, test_size=0.7, random_state=seed)
        else:
            train_idx, val_idx, test_idx = indices, indices, indices

        # Subsample train by label_fraction
        if 0.0 < label_fraction < 1.0:
            n_sub = max(1, int(len(train_idx) * label_fraction))
            train_idx = train_idx[:n_sub]

        train_ds = Subset(full_labeled_ds, train_idx)
        val_ds = Subset(full_labeled_ds, val_idx)
        test_ds = Subset(full_labeled_ds, test_idx)

        loaders["train"] = DataLoader(
            train_ds,
            batch_size=min(batch_size, max(1, len(train_ds))),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_mem,
        )
        loaders["val"] = DataLoader(
            val_ds,
            batch_size=min(batch_size, max(1, len(val_ds))),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_mem,
        )
        loaders["test"] = DataLoader(
            test_ds,
            batch_size=min(batch_size, max(1, len(test_ds))),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_mem,
        )

    return loaders

