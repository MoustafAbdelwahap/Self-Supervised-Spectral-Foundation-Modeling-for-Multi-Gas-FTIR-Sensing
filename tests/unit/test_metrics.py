"""Unit tests for sfm.evaluation.metrics module."""

import numpy as np
from sfm.evaluation.metrics import (
    compute_aggregate_metrics,
    compute_f1,
    compute_mae,
    compute_per_gas_metrics,
    compute_r2,
)


def test_r2_perfect():
    """Identical values must yield R² score of 1.0."""
    y = np.array([0.1, 0.5, 0.9, 1.2])
    score = compute_r2(y, y)
    assert np.isclose(score, 1.0)


def test_mae_zero():
    """Identical values must yield MAE of 0.0."""
    y = np.array([0.1, 0.5, 0.9, 1.2])
    score = compute_mae(y, y)
    assert np.isclose(score, 0.0)


def test_f1_perfect():
    """Identical binary classifications must yield macro F1 of 1.0."""
    y = np.array([0, 1, 1, 0, 1])
    score = compute_f1(y, y, threshold=0.5)
    assert np.isclose(score, 1.0)


def test_per_gas_metrics():
    """compute_per_gas_metrics must output dictionary with correct keys and ranges."""
    gas_names = ["CO2", "CH4", "CO"]

    y_true_class = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0],
    ] * 5)
    y_pred_class = np.array([
        [0.9, 0.1, 0.8],
        [0.2, 0.8, 0.1],
        [0.9, 0.9, 0.85],
        [0.1, 0.1, 0.2],
    ] * 5)

    y_true_conc = np.array([
        [0.5, 0.0, 0.3],
        [0.0, 0.4, 0.0],
        [0.6, 0.5, 0.4],
        [0.0, 0.0, 0.0],
    ] * 5)
    y_pred_conc = np.array([
        [0.51, 0.01, 0.29],
        [0.01, 0.39, 0.01],
        [0.59, 0.51, 0.41],
        [0.01, 0.01, 0.01],
    ] * 5)

    metrics = compute_per_gas_metrics(
        y_true_class, y_pred_class, y_true_conc, y_pred_conc, gas_names, threshold=0.5
    )

    for gas in gas_names:
        assert gas in metrics
        assert "r2" in metrics[gas]
        assert "mae" in metrics[gas]
        assert "f1" in metrics[gas]
        assert metrics[gas]["mae"] < 0.05
        assert metrics[gas]["f1"] == 1.0


def test_aggregate_metrics():
    """compute_aggregate_metrics must correctly compute the mean of per-gas values."""
    mock_per_gas = {
        "CO2": {"r2": 0.9, "mae": 0.1, "f1": 0.8},
        "CH4": {"r2": 0.8, "mae": 0.2, "f1": 0.9},
    }
    agg = compute_aggregate_metrics(mock_per_gas)
    assert np.isclose(agg["r2"], 0.85)
    assert np.isclose(agg["mae"], 0.15)
    assert np.isclose(agg["f1"], 0.85)
