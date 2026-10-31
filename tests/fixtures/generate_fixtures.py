"""Generate deterministic mini fixtures for fast unit and integration testing."""

import os
import numpy as np


def generate_all_fixtures(output_dir: str = None) -> None:
    """Generate mini spectra, labels, and wavenumbers files."""
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(42)

    # 1. mini_spectra.npy: (100, 200) float32 in [0, 1]
    mini_spectra = rng.uniform(0.0, 1.0, size=(100, 200)).astype(np.float32)
    spectra_path = os.path.join(output_dir, "mini_spectra.npy")
    np.save(spectra_path, mini_spectra)

    # 2. wavenumbers.npy: 200 points in [2000.0, 4000.0]
    wavenumbers = np.linspace(2000.0, 4000.0, 200, dtype=np.float32)
    wavenumbers_path = os.path.join(output_dir, "wavenumbers.npy")
    np.save(wavenumbers_path, wavenumbers)

    # 3. mini_labels.npz: 20 samples, 5 gases
    gas_names = np.array(["CO2", "CO", "CH4", "N2O", "H2O"])
    labeled_spectra = rng.uniform(0.0, 1.0, size=(20, 200)).astype(np.float32)
    gas_labels = rng.choice([0, 1], size=(20, 5), p=[0.6, 0.4]).astype(np.int32)
    concentrations = (rng.uniform(0.05, 1.0, size=(20, 5)) * gas_labels).astype(np.float32)

    labels_path = os.path.join(output_dir, "mini_labels.npz")
    np.savez(
        labels_path,
        spectra=labeled_spectra,
        gas_labels=gas_labels,
        concentrations=concentrations,
        gas_names=gas_names,
    )

    print(f"Test fixtures successfully generated in: {output_dir}")


if __name__ == "__main__":
    generate_all_fixtures()

