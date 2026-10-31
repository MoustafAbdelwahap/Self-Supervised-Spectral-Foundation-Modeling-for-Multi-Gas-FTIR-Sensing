"""Unit tests for classical baseline models (PLSR and Linear LS)."""

import os
import shutil
import numpy as np
import pytest

from sfm.evaluation.baseline import (
    load_baseline,
    predict_plsr,
    train_and_save_baseline,
    train_plsr,
)


@pytest.fixture
def mini_data():
    """Load mini fixtures for testing."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), "..", "fixtures")
    spectra = np.load(os.path.join(fixtures_dir, "mini_spectra.npy"))[:20]
    data = np.load(os.path.join(fixtures_dir, "mini_labels.npz"))
    concentrations = data["concentrations"]
    return spectra, concentrations


def test_train_plsr_shapes(mini_data):
    """Verify PLSR training and prediction shapes."""
    spectra, concentrations = mini_data
    n_samples, n_gases = concentrations.shape

    model = train_plsr(spectra, concentrations, n_components=5)
    class_preds, conc_preds = predict_plsr(model, spectra)

    assert class_preds.shape == (n_samples, n_gases)
    assert conc_preds.shape == (n_samples, n_gases)
    assert (conc_preds >= 0.0).all()
    assert set(np.unique(class_preds)).issubset({0, 1})


def test_n_components_clamping(mini_data):
    """Verify n_components is safely clamped when exceeding sample count."""
    spectra, concentrations = mini_data
    # Requesting 100 components on 20 samples
    model = train_plsr(spectra, concentrations, n_components=100)
    assert model.n_components <= len(spectra) - 1


def test_train_and_save_baseline(tmp_path):
    """Verify end-to-end saving and reloading of baseline model."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), "..", "fixtures")
    data_dir = str(tmp_path / "data")
    output_path = str(tmp_path / "baseline_plsr.pkl")
    os.makedirs(data_dir, exist_ok=True)

    shutil.copy(os.path.join(fixtures_dir, "mini_labels.npz"), os.path.join(data_dir, "mini_labels.npz"))
    shutil.copy(os.path.join(fixtures_dir, "wavenumbers.npy"), os.path.join(data_dir, "wavenumbers.npy"))

    saved_path = train_and_save_baseline(
        data_dir=data_dir,
        method="plsr",
        n_components=3,
        output_path=output_path,
        seed=42,
    )
    assert os.path.exists(saved_path)

    reloaded = load_baseline(saved_path)
    sample_spectra = np.random.randn(2, 200).astype(np.float32)
    cls_preds, conc_preds = predict_plsr(reloaded, sample_spectra)
    assert cls_preds.shape == (2, 5)
    assert conc_preds.shape == (2, 5)


def test_linear_ls_baseline(tmp_path):
    """Verify Linear LS baseline training and saving."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), "..", "fixtures")
    data_dir = str(tmp_path / "data")
    output_path = str(tmp_path / "baseline_ridge.pkl")
    os.makedirs(data_dir, exist_ok=True)

    shutil.copy(os.path.join(fixtures_dir, "mini_labels.npz"), os.path.join(data_dir, "mini_labels.npz"))
    shutil.copy(os.path.join(fixtures_dir, "wavenumbers.npy"), os.path.join(data_dir, "wavenumbers.npy"))

    saved_path = train_and_save_baseline(
        data_dir=data_dir,
        method="linear_ls",
        output_path=output_path,
        seed=42,
    )
    assert os.path.exists(saved_path)
    reloaded = load_baseline(saved_path)
    sample_spectra = np.random.randn(2, 200).astype(np.float32)
    cls_preds, conc_preds = predict_plsr(reloaded, sample_spectra)
    assert conc_preds.shape == (2, 5)

