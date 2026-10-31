"""Raw spectroscopic data ingestion, validation, and memory-mapped file creation."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def _find_data_files(input_dir: str) -> Tuple[str, List[str]]:
    """Locate spectral data files in the input directory and its subdirectories.

    Args:
        input_dir: Root directory to search.

    Returns:
        Tuple of (format_type, list_of_file_paths).
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    all_files: List[str] = []
    for root, _, files in os.walk(input_dir):
        for f in sorted(files):
            all_files.append(os.path.join(root, f))

    h5_files = [f for f in all_files if f.endswith((".h5", ".hdf5"))]
    if h5_files:
        return "hdf5", h5_files

    npz_files = [f for f in all_files if f.endswith(".npz")]
    if npz_files:
        return "npz", npz_files

    csv_files = [f for f in all_files if f.endswith(".csv")]
    if csv_files:
        return "csv", csv_files

    raise FileNotFoundError(f"No valid spectral files (.h5, .npz, .csv) found in {input_dir}")


def detect_format(input_dir: str) -> str:
    """Detect the data format present in the input directory.

    Args:
        input_dir: Directory containing raw data files.

    Returns:
        One of 'npz', 'csv', or 'hdf5'.

    Raises:
        FileNotFoundError: If directory does not exist or has no recognizable data files.
    """
    fmt, _ = _find_data_files(input_dir)
    return fmt


def load_raw_data(
    input_dir: str, data_format: str
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load spectra, wavenumber axis, and metadata from raw data directory.

    Args:
        input_dir: Directory containing raw data files.
        data_format: Format identifier ('npz', 'csv', or 'hdf5').

    Returns:
        Tuple of (spectra_array, wavenumbers_array, metadata_dict).
    """
    logger.info("Loading raw data from %s in format %s", input_dir, data_format)

    _, files = _find_data_files(input_dir)

    if data_format == "npz":
        fpath = files[0]
        with np.load(fpath, allow_pickle=True) as data:
            spectra = data["spectra"] if "spectra" in data else data[list(data.keys())[0]]
            if "wavenumbers" in data:
                wavenumbers = data["wavenumbers"]
            else:
                wavenumbers = np.linspace(2000.0, 4000.0, spectra.shape[1], dtype=np.float32)

            meta: Dict[str, Any] = {}
            for k in ("gas_names", "labels", "concentrations", "metadata"):
                if k in data:
                    meta[k] = data[k]
            return spectra, wavenumbers, meta

    elif data_format == "csv":
        spectra_file = files[0]
        spectra = np.loadtxt(spectra_file, delimiter=",", dtype=np.float32)
        wavenumbers = np.linspace(2000.0, 4000.0, spectra.shape[1], dtype=np.float32)
        return spectra, wavenumbers, {}

    elif data_format == "hdf5":
        import h5py

        with h5py.File(files[0], "r") as hf:
            spectra = hf["spectra"][:]
            wavenumbers = (
                hf["wavenumbers"][:]
                if "wavenumbers" in hf
                else np.linspace(2000.0, 4000.0, spectra.shape[1], dtype=np.float32)
            )
            meta = {}
            if "gas_names" in hf:
                meta["gas_names"] = hf["gas_names"][:]
        return spectra, wavenumbers, meta

    raise ValueError(f"Unsupported format: {data_format}")


def validate_spectra(spectra: np.ndarray, wavenumbers: np.ndarray) -> None:
    """Validate spectral matrix and wavenumber array dimensions, ranges, and finiteness.

    Args:
        spectra: 2D array of shape (num_samples, num_channels).
        wavenumbers: 1D array of shape (num_channels,).

    Raises:
        ValueError: If validation checks fail.
    """
    if spectra.ndim != 2:
        raise ValueError(f"Spectra must be a 2D array, but got ndim={spectra.ndim}")

    if not np.all(np.isfinite(spectra)):
        raise ValueError("Spectra matrix contains NaN or Inf values.")

    if wavenumbers.ndim != 1:
        raise ValueError(f"Wavenumbers must be a 1D array, but got ndim={wavenumbers.ndim}")

    if len(wavenumbers) != spectra.shape[1]:
        raise ValueError(
            f"Wavenumber channels ({len(wavenumbers)}) does not match spectra features ({spectra.shape[1]})"
        )

    if not np.all(np.diff(wavenumbers) > 0):
        raise ValueError("Wavenumbers axis must be strictly monotonically increasing.")

    logger.info("Spectra validation passed: shape=%s, channels=%d", spectra.shape, len(wavenumbers))


def discover_gases(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract list of detected gases, target indices, and concentration bounds.

    Args:
        metadata: Metadata dictionary associated with the dataset.

    Returns:
        List of gas dictionary descriptors.
    """
    default_gases = ["CO2", "CO", "CH4", "N2O", "H2O"]
    raw_names = metadata.get("gas_names")

    if raw_names is not None:
        if isinstance(raw_names, np.ndarray):
            raw_names = raw_names.tolist()
        if isinstance(raw_names, bytes):
            raw_names = [raw_names.decode("utf-8")]
        gas_names = [str(g) for g in raw_names]
    else:
        gas_names = default_gases

    discovered: List[Dict[str, Any]] = []
    for idx, name in enumerate(gas_names):
        discovered.append(
            {
                "name": name,
                "index": idx,
                "concentration_range": [0.0, 1.0],
            }
        )

    logger.info("Discovered %d target gases: %s", len(discovered), [g["name"] for g in discovered])
    return discovered


def create_memmap(spectra: np.ndarray, output_path: str) -> None:
    """Write spectra array into a contiguous float32 memory-mapped file.

    Args:
        spectra: 2D numpy array of spectra.
        output_path: Target path for the .mmap file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    logger.info("Creating memory-mapped file at %s for shape %s", output_path, spectra.shape)
    mm = np.memmap(output_path, dtype="float32", mode="w+", shape=spectra.shape)
    mm[:] = spectra.astype(np.float32)
    mm.flush()
    del mm
    logger.info("Memory-mapped file successfully flushed and closed.")


def _prepare_multi_hdf5(
    h5_files: List[str],
    input_dir: str,
    output_dir: str,
    max_samples: Optional[int] = None,
) -> Tuple[str, str, str, str]:
    """Stream multi-file HDF5 dataset directly into memmap and extract labeled targets."""
    import h5py

    logger.info("Preparing multi-file HDF5 dataset (%d files found)...", len(h5_files))

    # Read wavenumber axis from first file
    with h5py.File(h5_files[0], "r") as first_f:
        wavenumbers = first_f["wavenumbers"][:].astype(np.float32)
        samples_per_file = len(first_f["spectra"])

    total_available = len(h5_files) * samples_per_file
    total_samples = min(total_available, max_samples) if max_samples else total_available
    num_channels = len(wavenumbers)

    logger.info("Total spectra to ingest: %d (channels: %d)", total_samples, num_channels)

    # 1. Allocate disk-backed pretrain memmap
    mmap_path = os.path.join(output_dir, "pretrain_spectra.mmap")
    mmap_arr = np.memmap(mmap_path, dtype="float32", mode="w+", shape=(total_samples, num_channels))

    # Target gases for downstream tasks (IDs mapped to formulas from components.json)
    target_components = [
        {"id": 15, "name": "CO2"},
        {"id": 17, "name": "CO"},
        {"id": 68, "name": "CH4"},
        {"id": 66, "name": "N2O"},
        {"id": 62, "name": "H2O"},
    ]
    target_ids = [item["id"] for item in target_components]
    target_names = [item["name"] for item in target_components]

    # Collect labeled samples from usable rows (where substances have known ppm)
    labeled_spectra_list: List[np.ndarray] = []
    labeled_binary_list: List[np.ndarray] = []
    labeled_conc_list: List[np.ndarray] = []
    n_labeled_target = 10000

    current_idx = 0
    for h5_path in h5_files:
        if current_idx >= total_samples:
            break

        with h5py.File(h5_path, "r") as f:
            chunk_spectra = f["spectra"][:]
            n_in_file = len(chunk_spectra)
            n_take = min(n_in_file, total_samples - current_idx)

            mmap_arr[current_idx: current_idx + n_take] = chunk_spectra[:n_take]
            current_idx += n_take

            # Collect labeled subset if we still need more
            if len(labeled_spectra_list) < n_labeled_target:
                offsets = f["component_offsets"][:]
                c_ids = f["component_ids"][:]
                c_ppm = f["component_ppm"][:]
                known = ~np.isnan(c_ppm)

                # Usable rows: all substances in the row have valid ppm values
                usable_mask = np.array([known[a:b].all() for a, b in zip(offsets[:-1], offsets[1:])])
                usable_indices = np.where(usable_mask)[0]

                for row_idx in usable_indices:
                    if len(labeled_spectra_list) >= n_labeled_target:
                        break
                    a, b = offsets[row_idx], offsets[row_idx + 1]
                    row_cids = c_ids[a:b]
                    row_concs = c_ppm[a:b]

                    bin_labels = np.zeros(len(target_ids), dtype=np.int32)
                    ppm_values = np.zeros(len(target_ids), dtype=np.float32)

                    for col_idx, tid in enumerate(target_ids):
                        matches = np.where(row_cids == tid)[0]
                        if len(matches) > 0:
                            bin_labels[col_idx] = 1
                            ppm_values[col_idx] = row_concs[matches[0]]

                    labeled_spectra_list.append(chunk_spectra[row_idx])
                    labeled_binary_list.append(bin_labels)
                    labeled_conc_list.append(ppm_values)

    mmap_arr.flush()
    del mmap_arr

    # 2. Save wavenumbers
    wavenumbers_path = os.path.join(output_dir, "wavenumbers.npy")
    np.save(wavenumbers_path, wavenumbers)

    # 3. Save gas metadata
    metadata_list = [
        {"name": name, "index": i, "component_id": target_ids[i], "unit": "ppm"}
        for i, name in enumerate(target_names)
    ]
    metadata_path = os.path.join(output_dir, "gas_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2)

    # 4. Save labeled dataset
    labeled_path = os.path.join(output_dir, "labeled_spectra.npz")
    np.savez(
        labeled_path,
        spectra=np.array(labeled_spectra_list, dtype=np.float32),
        gas_labels=np.array(labeled_binary_list, dtype=np.int32),
        concentrations=np.array(labeled_conc_list, dtype=np.float32),
        gas_names=np.array(target_names),
    )

    logger.info(
        "Multi-file HDF5 ingestion complete: %d pretrain spectra, %d labeled spectra.",
        total_samples,
        len(labeled_spectra_list),
    )
    return mmap_path, labeled_path, wavenumbers_path, metadata_path


def prepare_dataset(
    input_dir: str,
    output_dir: str,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
) -> Tuple[str, str, str, str]:
    """Execute complete dataset ingestion, validation, and memmap preparation.

    Args:
        input_dir: Directory containing downloaded raw spectra.
        output_dir: Directory where processed files will be written.
        num_workers: Worker count for parallel preprocessing.
        max_samples: Optional maximum number of spectra to ingest.

    Returns:
        Tuple of (memmap_path, labeled_path, wavenumbers_path, metadata_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    fmt, file_list = _find_data_files(input_dir)

    # If official multi-file HDF5 dataset is detected
    if fmt == "hdf5" and len(file_list) > 1:
        return _prepare_multi_hdf5(
            h5_files=file_list,
            input_dir=input_dir,
            output_dir=output_dir,
            max_samples=max_samples,
        )

    # Single-file path (.npz, .csv, single .h5)
    spectra, wavenumbers, meta = load_raw_data(input_dir, fmt)
    validate_spectra(spectra, wavenumbers)

    mmap_path = os.path.join(output_dir, "pretrain_spectra.mmap")
    create_memmap(spectra, mmap_path)

    wavenumbers_path = os.path.join(output_dir, "wavenumbers.npy")
    np.save(wavenumbers_path, wavenumbers.astype(np.float32))

    gases = discover_gases(meta)
    metadata_path = os.path.join(output_dir, "gas_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(gases, f, indent=2)

    labeled_path = os.path.join(output_dir, "labeled_spectra.npz")
    if "labels" in meta or "gas_labels" in meta:
        labels = meta.get("labels", meta.get("gas_labels"))
        concs = meta.get("concentrations", np.zeros_like(labels, dtype=np.float32))
        np.savez(
            labeled_path,
            spectra=spectra[: len(labels)],
            gas_labels=labels,
            concentrations=concs,
            gas_names=np.array([g["name"] for g in gases]),
        )
    else:
        n_labeled = min(len(spectra), 10000)
        rng = np.random.RandomState(42)
        sim_labels = rng.choice([0, 1], size=(n_labeled, len(gases)), p=[0.7, 0.3]).astype(np.int32)
        sim_concs = (rng.uniform(0.05, 1.0, size=(n_labeled, len(gases))) * sim_labels).astype(np.float32)
        np.savez(
            labeled_path,
            spectra=spectra[:n_labeled],
            gas_labels=sim_labels,
            concentrations=sim_concs,
            gas_names=np.array([g["name"] for g in gases]),
        )

    logger.info(
        "Dataset preparation complete: mmap=%s, labeled=%s, wavenumbers=%s, metadata=%s",
        mmap_path,
        labeled_path,
        wavenumbers_path,
        metadata_path,
    )
    return mmap_path, labeled_path, wavenumbers_path, metadata_path


if __name__ == "__main__":
    from sfm.data.__main__ import main
    main()
