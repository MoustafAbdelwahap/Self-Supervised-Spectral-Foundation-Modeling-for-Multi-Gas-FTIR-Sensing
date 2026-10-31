"""Unit tests for HITRAN data simulation, ILS convolution, and synthetic dataset generation."""

import os
import numpy as np
import pytest
from sfm.data.hitran import (
    add_perturbations,
    apply_ils,
    beer_lambert_mix,
    fetch_hitran_crosssections,
    generate_dataset,
    generate_synthetic_spectrum,
)


def test_fetch_hitran_import_error_message():
    """Verify fetch_hitran_crosssections raises ImportError with installation instructions if hapi missing."""
    with pytest.raises(ImportError) as exc_info:
        fetch_hitran_crosssections("CO2", 2000.0, 4000.0)
    assert "pip install hitran-api" in str(exc_info.value)


def test_apply_ils_shape_and_range():
    """Verify sinc ILS convolution preserves array length and non-negativity."""
    spectrum = np.zeros(200, dtype=np.float32)
    spectrum[100] = 5.0  # Sharp delta peak
    ils_spectrum = apply_ils(spectrum, resolution_cm=40.0, delta_wavenumber=10.0)

    assert ils_spectrum.shape == (200,)
    assert (ils_spectrum >= 0.0).all()
    # Peak should be broadened
    assert ils_spectrum[100] < 5.0
    assert ils_spectrum[99] > 0.0
    assert ils_spectrum[101] > 0.0


def test_beer_lambert_mix():
    """Verify linear superposition of gas cross-sections according to Beer-Lambert law."""
    sigma1 = np.ones(50, dtype=np.float32) * 2.0
    sigma2 = np.ones(50, dtype=np.float32) * 3.0
    crosssections = {"GAS_A": sigma1, "GAS_B": sigma2}

    mix = beer_lambert_mix(crosssections, {"GAS_A": 0.5, "GAS_B": 0.2}, path_length=1.0)
    expected = 0.5 * 2.0 + 0.2 * 3.0  # 1.0 + 0.6 = 1.6
    np.testing.assert_allclose(mix, expected, rtol=1e-5)


def test_add_perturbations():
    """Verify sensor perturbations add realistic noise and baseline variation."""
    pure = np.ones(100, dtype=np.float32)
    perturbed = add_perturbations(pure, snr=500.0, drift_amplitude=0.02, seed=42)

    assert perturbed.shape == (100,)
    assert not np.allclose(perturbed, pure)
    assert (perturbed >= 0.0).all()


def test_generate_synthetic_spectrum():
    """Verify single spectrum synthesis returns expected spectrum array and metadata."""
    crosssections = {
        "CO2": np.ones(100, dtype=np.float32),
        "CH4": np.ones(100, dtype=np.float32) * 0.5,
    }
    spectrum, meta = generate_synthetic_spectrum(
        crosssections,
        n_gases_range=(1, 2),
        concentration_range=(0.1, 0.9),
        seed=123,
    )

    assert spectrum.shape == (100,)
    assert "gases" in meta
    assert "concentrations" in meta
    assert "snr" in meta
    assert "drift" in meta
    assert len(meta["gases"]) >= 1


def test_generate_dataset_mini(tmp_path):
    """Verify batch synthetic dataset generator produces all required files and shapes."""
    out_dir = str(tmp_path / "synthetic")
    mmap_p, labeled_p, wn_p, meta_p = generate_dataset(
        output_dir=out_dir,
        num_samples=25,
        gas_names=["CO2", "CH4"],
        wavenumber_range=(2000.0, 2500.0),
        resolution_cm=40.0,
        seed=42,
    )

    assert os.path.exists(mmap_p)
    assert os.path.exists(labeled_p)
    assert os.path.exists(wn_p)
    assert os.path.exists(meta_p)

    # Check memmap shape
    wn = np.load(wn_p)
    n_channels = len(wn)
    mm = np.memmap(mmap_p, dtype="float32", mode="r", shape=(25, n_channels))
    assert mm.shape == (25, n_channels)
    assert not np.isnan(mm).any()

    # Check labeled NPZ
    npz = np.load(labeled_p)
    assert npz["spectra"].shape == (25, n_channels)
    assert npz["gas_labels"].shape == (25, 2)
    assert npz["concentrations"].shape == (25, 2)

