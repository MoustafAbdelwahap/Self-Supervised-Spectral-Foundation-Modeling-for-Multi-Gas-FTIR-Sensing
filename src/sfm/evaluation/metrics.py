"""Quantitative performance evaluation metrics for multi-gas identification and regression."""

from typing import Dict, List
import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, r2_score


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² (coefficient of determination) regression score.

    Args:
        y_true: Ground-truth concentration targets.
        y_pred: Predicted concentration values.

    Returns:
        Float R² score.
    """
    return float(r2_score(y_true, y_pred))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error (MAE) score.

    Args:
        y_true: Ground-truth concentration targets.
        y_pred: Predicted concentration values.

    Returns:
        Float MAE score.
    """
    return float(mean_absolute_error(y_true, y_pred))


def compute_f1(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
) -> float:
    """Calculate macro-averaged F1 score for binary or multi-label classification.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted probabilities or binary values.
        threshold: Decision threshold for binarization (default: 0.5).

    Returns:
        Float macro-averaged F1 score.
    """
    y_pred_bin = (np.asarray(y_pred) >= threshold).astype(int)
    y_true_bin = (np.asarray(y_true) >= 0.5).astype(int)
    return float(f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0))


def compute_per_gas_metrics(
    y_true_class: np.ndarray,
    y_pred_class: np.ndarray,
    y_true_conc: np.ndarray,
    y_pred_conc: np.ndarray,
    gas_names: List[str],
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Compute independent R², MAE, and F1 metrics for each identified gas.

    Args:
        y_true_class: Ground truth binary matrix of shape (samples, gases).
        y_pred_class: Predicted class probabilities or logits of shape (samples, gases).
        y_true_conc: Ground truth concentrations of shape (samples, gases).
        y_pred_conc: Predicted concentrations of shape (samples, gases).
        gas_names: List of target gas identifiers matching column indices.
        threshold: Threshold for gas presence classification.

    Returns:
        Dictionary mapping each gas name to its specific evaluation metrics:
        {"GasName": {"r2": float, "mae": float, "f1": float}, ...}
    """
    y_true_c = np.asarray(y_true_class)
    y_pred_c = np.asarray(y_pred_class)
    y_true_r = np.asarray(y_true_conc)
    y_pred_r = np.asarray(y_pred_conc)

    # Convert logits/probabilities to binary if continuous
    y_pred_c_bin = (y_pred_c >= threshold).astype(int)
    y_true_c_bin = (y_true_c >= 0.5).astype(int)

    results: Dict[str, Dict[str, float]] = {}
    for i, gas in enumerate(gas_names):
        col_true_c = y_true_c_bin[:, i]
        col_pred_c = y_pred_c_bin[:, i]
        col_true_r = y_true_r[:, i]
        col_pred_r = y_pred_r[:, i]

        r2_val = float(r2_score(col_true_r, col_pred_r))
        mae_val = float(mean_absolute_error(col_true_r, col_pred_r))

        if (col_true_c == 0).all() and (col_pred_c == 0).all():
            f1_val = 1.0
        elif (col_true_c == 1).all() and (col_pred_c == 1).all():
            f1_val = 1.0
        else:
            f1_val = float(f1_score(col_true_c, col_pred_c, zero_division=0))

        results[gas] = {
            "r2": r2_val,
            "mae": mae_val,
            "f1": f1_val,
        }

    return results


def compute_aggregate_metrics(per_gas: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Calculate mean aggregate evaluation metrics across all evaluated gases.

    Args:
        per_gas: Dictionary output from compute_per_gas_metrics.

    Returns:
        Dictionary containing mean R², mean MAE, and mean F1 across all gases.
    """
    if not per_gas:
        return {"r2": 0.0, "mae": 0.0, "f1": 0.0}

    r2_vals = [m["r2"] for m in per_gas.values()]
    mae_vals = [m["mae"] for m in per_gas.values()]
    f1_vals = [m["f1"] for m in per_gas.values()]

    mean_r2 = float(np.mean(r2_vals))
    mean_mae = float(np.mean(mae_vals))
    mean_f1 = float(np.mean(f1_vals))

    return {
        "r2": mean_r2,
        "mae": mean_mae,
        "f1": mean_f1,
        "mean_r2": mean_r2,
        "mean_mae": mean_mae,
        "mean_f1": mean_f1,
    }
