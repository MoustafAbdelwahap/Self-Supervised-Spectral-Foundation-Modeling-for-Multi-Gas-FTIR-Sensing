"""Classical spectral baseline models: Partial Least Squares Regression (PLSR) and Linear LS.

Provides:
- train_plsr: Fit PLSR with adaptive component count clamping.
- predict_plsr: Predict continuous concentrations and binary presence.
- train_and_save_baseline: Orchestrate baseline model training and pickling.
- load_baseline: Load serialized baseline model.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Optional, Tuple

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from sfm.data.dataset import create_dataloaders

logger = logging.getLogger(__name__)


def train_plsr(
    spectra: np.ndarray,
    concentrations: np.ndarray,
    n_components: int = 20,
) -> PLSRegression:
    """Fit a Partial Least Squares Regression (PLSR) model.

    Args:
        spectra: Array of absorption spectra of shape (n_samples, n_channels).
        concentrations: Array of target concentrations of shape (n_samples, n_gases).
        n_components: Requested number of latent components (default: 20).

    Returns:
        Fitted PLSRegression model instance.
    """
    n_samples, n_features = spectra.shape
    max_components = min(n_samples - 1, n_features)
    if max_components < 1:
        max_components = 1

    if n_components > max_components:
        logger.warning(
            "Requested n_components (%d) exceeds sample/feature limit (%d). Clamping to %d.",
            n_components,
            max_components,
            max_components,
        )
        n_components = max_components

    model = PLSRegression(n_components=n_components, scale=True)
    model.fit(spectra, concentrations)
    logger.info("Fitted PLSRegression with %d components on %d samples.", n_components, n_samples)
    return model


def predict_plsr(
    model: Any,
    spectra: np.ndarray,
    threshold: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate concentration and presence predictions from a fitted baseline model.

    Args:
        model: Fitted PLSRegression or Linear LS model.
        spectra: Array of absorption spectra of shape (n_samples, n_channels).
        threshold: Concentration threshold for positive gas presence.

    Returns:
        Tuple of:
        - class_predictions: Binary presence indicator matrix of shape (n_samples, n_gases).
        - concentration_predictions: Continuous concentration values of shape (n_samples, n_gases).
    """
    concentrations = model.predict(spectra)
    # Clip negative concentration predictions to 0.0 (concentrations are non-negative)
    concentrations = np.maximum(concentrations, 0.0)
    class_preds = (concentrations >= threshold).astype(int)
    return class_preds, concentrations


def train_and_save_baseline(
    data_dir: str,
    method: str = "plsr",
    n_components: int = 20,
    output_path: Optional[str] = None,
    seed: int = 42,
) -> str:
    """Train classical baseline on training spectra and serialize to disk.

    Args:
        data_dir: Directory containing processed spectral data.
        method: Baseline algorithm ('plsr' or 'linear_ls').
        n_components: Number of latent components for PLSR.
        output_path: Destination path for pickled model file.
        seed: Random seed for data splitting reproducibility.

    Returns:
        Filepath to the saved model pickle file.
    """
    if output_path is None:
        output_path = os.path.join(data_dir, f"baseline_{method}.pkl")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Load labeled training dataset via standard data loader split
    loaders = create_dataloaders(
        data_dir=data_dir,
        batch_size=256,
        num_workers=0,
        label_fraction=1.0,
        seed=seed,
    )
    train_loader = loaders.get("train")
    if train_loader is None or len(train_loader.dataset) == 0:
        raise ValueError(f"No labeled training samples found in {data_dir}")

    all_spectra = []
    all_concs = []
    for spectra_b, _, conc_b in train_loader:
        all_spectra.append(spectra_b.numpy())
        all_concs.append(conc_b.numpy())

    train_spectra = np.concatenate(all_spectra, axis=0)
    train_concs = np.concatenate(all_concs, axis=0)

    method = method.lower().strip()
    if method == "plsr":
        model = train_plsr(train_spectra, train_concs, n_components=n_components)
    elif method == "linear_ls":
        model = Ridge(alpha=1.0, random_state=seed)
        model.fit(train_spectra, train_concs)
        logger.info("Fitted Ridge Linear LS baseline on %d samples.", len(train_spectra))
    else:
        raise ValueError(f"Unknown baseline method '{method}'. Choose 'plsr' or 'linear_ls'.")

    # Evaluate training fit
    train_preds = model.predict(train_spectra)
    r2_train = r2_score(train_concs, train_preds, multioutput="uniform_average")
    logger.info("Baseline (%s) Training R²: %.4f", method, r2_train)

    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Baseline model saved to %s", output_path)

    return output_path


def load_baseline(path: str) -> Any:
    """Load pickled baseline model.

    Args:
        path: Filepath to serialized model pickle file.

    Returns:
        Deserialized sklearn model instance.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Baseline model not found at: {path}")

    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info("Loaded baseline model from %s", path)
    return model

