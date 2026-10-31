"""Benchmarking and evaluation runner comparing Spectral Foundation Model (SFM) vs. PLSR baseline.

Computes multi-gas quantification R², MAE, detection F1, and Out-of-Distribution (OOD)
robustness under sensor noise amplification and optical baseline drift.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from sfm.data.dataset import create_dataloaders, get_conc_scales
from sfm.evaluation.baseline import load_baseline, predict_plsr
from sfm.evaluation.metrics import compute_aggregate_metrics, compute_per_gas_metrics
from sfm.models.heads import create_downstream_model
from sfm.training.finetune import evaluate
from sfm.training.utils import get_device, set_seed

logger = logging.getLogger(__name__)


def _discover_gas_names(data_dir: str) -> List[str]:
    """Discover gas identifiers from metadata JSON or dataset files."""
    meta_path = os.path.join(data_dir, "gas_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                if isinstance(meta, list):
                    # Support both "name" and legacy "gas" keys
                    names = []
                    for item in meta:
                        if isinstance(item, dict):
                            names.append(item.get("name", item.get("gas", str(item))))
                        else:
                            names.append(str(item))
                    return names
                elif isinstance(meta, dict):
                    return meta.get("gas_names", list(meta.keys()))
        except Exception as e:
            logger.warning("Could not read gas_metadata.json: %s", e)

    for fname in ("labeled_spectra.npz", "mini_labels.npz"):
        npz_path = os.path.join(data_dir, fname)
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            if "gas_names" in data:
                return list(data["gas_names"])
            elif "gas_labels" in data:
                return [f"Gas_{i}" for i in range(data["gas_labels"].shape[1])]

    return ["Gas_0", "Gas_1", "Gas_2"]


def evaluate_sfm(
    checkpoint_path: str,
    data_dir: str,
    device: torch.device,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate fine-tuned SFM downstream model on standard test set.

    Args:
        checkpoint_path: Path to fine-tuned downstream checkpoint (.pt).
        data_dir: Directory containing processed dataset.
        device: Active PyTorch compute device.
        seed: Random split seed.

    Returns:
        Evaluation metrics dict with 'loss', 'per_gas', and 'aggregate'.
    """
    gas_names = _discover_gas_names(data_dir)
    model = create_downstream_model(
        pretrained_checkpoint_path=checkpoint_path,
        head_config={"hidden_dim": 128, "dropout": 0.0},
        num_gases=len(gas_names),
        device=device,
    )
    loaders = create_dataloaders(
        data_dir=data_dir,
        batch_size=128,
        num_workers=0,
        label_fraction=1.0,
        seed=seed,
    )
    test_loader = loaders.get("test")
    if test_loader is None or len(test_loader.dataset) == 0:
        raise ValueError(f"No test samples found in {data_dir}")

    metrics = evaluate(model=model, dataloader=test_loader, device=device, gas_names=gas_names)
    logger.info(
        "SFM Test Evaluation: Mean R²=%.4f, Mean F1=%.4f",
        metrics["aggregate"].get("mean_r2", 0.0),
        metrics["aggregate"].get("mean_f1", 0.0),
    )
    return metrics


def evaluate_baseline(
    model_path: str,
    data_dir: str,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate serialized baseline model (PLSR or Linear LS) on standard test set.

    Args:
        model_path: Path to serialized pickle model.
        data_dir: Directory containing processed dataset.
        seed: Random split seed.

    Returns:
        Evaluation metrics dict with 'per_gas' and 'aggregate'.
    """
    model = load_baseline(model_path)
    gas_names = _discover_gas_names(data_dir)

    loaders = create_dataloaders(
        data_dir=data_dir,
        batch_size=256,
        num_workers=0,
        label_fraction=1.0,
        seed=seed,
    )
    test_loader = loaders.get("test")
    if test_loader is None or len(test_loader.dataset) == 0:
        raise ValueError(f"No test samples found in {data_dir}")

    all_spectra = []
    all_labels = []
    all_concs = []
    for s, l, c in test_loader:
        all_spectra.append(s.numpy())
        all_labels.append(l.numpy())
        all_concs.append(c.numpy())

    test_spectra = np.concatenate(all_spectra, axis=0)
    test_labels = np.concatenate(all_labels, axis=0)
    test_concs = np.concatenate(all_concs, axis=0)

    class_preds, conc_preds = predict_plsr(model, test_spectra)

    conc_scales = get_conc_scales(data_dir)
    per_gas = compute_per_gas_metrics(
        y_true_class=test_labels,
        y_pred_class=class_preds,
        y_true_conc=test_concs * conc_scales,
        y_pred_conc=conc_preds * conc_scales,
        gas_names=gas_names,
    )
    aggregate = compute_aggregate_metrics(per_gas)

    logger.info(
        "Baseline Test Evaluation: Mean R²=%.4f, Mean F1=%.4f",
        aggregate.get("mean_r2", 0.0),
        aggregate.get("mean_f1", 0.0),
    )
    return {"per_gas": per_gas, "aggregate": aggregate}


def generate_ood_test_data(
    data_dir: str,
    noise_factor: float = 2.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate Out-of-Distribution (OOD) test set with amplified noise and baseline drift.

    Args:
        data_dir: Directory containing processed dataset.
        noise_factor: Multiplier for sensor Gaussian noise (default: 2.0).
        seed: Random seed for deterministic perturbation.

    Returns:
        Tuple of (ood_spectra, test_labels, test_concentrations).
    """
    loaders = create_dataloaders(
        data_dir=data_dir,
        batch_size=256,
        num_workers=0,
        label_fraction=1.0,
        seed=seed,
    )
    test_loader = loaders.get("test")
    if test_loader is None or len(test_loader.dataset) == 0:
        raise ValueError(f"No test samples found in {data_dir}")

    all_spectra = []
    all_labels = []
    all_concs = []
    for s, l, c in test_loader:
        all_spectra.append(s.numpy())
        all_labels.append(l.numpy())
        all_concs.append(c.numpy())

    test_spectra = np.concatenate(all_spectra, axis=0)
    test_labels = np.concatenate(all_labels, axis=0)
    test_concs = np.concatenate(all_concs, axis=0)

    rng = np.random.RandomState(seed)
    n_samples, n_channels = test_spectra.shape

    # 1. Amplified Gaussian noise perturbation
    spec_stds = np.std(test_spectra, axis=1, keepdims=True) + 1e-6
    noise_std = spec_stds * (noise_factor * 0.05)
    noise = rng.normal(0.0, noise_std, size=test_spectra.shape)

    # 2. Sinusoidal optical baseline drift
    x = np.linspace(0.0, 1.0, n_channels)
    drift = np.zeros_like(test_spectra)
    for i in range(n_samples):
        amp = rng.uniform(0.05, 0.15) * (np.max(test_spectra[i]) + 1e-4)
        freq = rng.uniform(1.0, 3.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        drift[i] = amp * np.sin(2.0 * np.pi * freq * x + phase)

    ood_spectra = np.clip(test_spectra + noise + drift, 0.0, None).astype(np.float32)
    logger.info("Generated %d OOD test spectra (noise_factor=%.2f, drift applied).", n_samples, noise_factor)
    return ood_spectra, test_labels, test_concs


def evaluate_robustness(
    sfm_checkpoint: str,
    baseline_path: str,
    data_dir: str,
    noise_factor: float,
    device: torch.device,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate both SFM and Baseline models on Out-of-Distribution test perturbations.

    Args:
        sfm_checkpoint: Path to fine-tuned downstream SFM checkpoint.
        baseline_path: Path to serialized baseline model pickle.
        data_dir: Directory containing processed data.
        noise_factor: OOD noise multiplier.
        device: Active PyTorch compute device.
        seed: Random seed.

    Returns:
        Dictionary containing standard metrics, OOD metrics, and robustness degradation deltas.
    """
    # 1. Standard In-Distribution Evaluation
    sfm_std = evaluate_sfm(sfm_checkpoint, data_dir, device, seed)
    base_std = evaluate_baseline(baseline_path, data_dir, seed)

    # 2. Generate OOD Test Data
    ood_spectra, test_labels, test_concs = generate_ood_test_data(data_dir, noise_factor=noise_factor, seed=seed)
    gas_names = _discover_gas_names(data_dir)

    # 3. SFM on OOD
    sfm_model = create_downstream_model(
        pretrained_checkpoint_path=sfm_checkpoint,
        head_config={"hidden_dim": 128, "dropout": 0.0},
        num_gases=len(gas_names),
        device=device,
    )
    sfm_model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(ood_spectra).to(device)
        class_logits, conc_preds = sfm_model(x_tensor)
        sfm_pred_cls = (torch.sigmoid(class_logits).cpu().numpy() >= 0.5).astype(int)
        sfm_pred_conc = conc_preds.cpu().numpy()

    conc_scales = get_conc_scales(data_dir)
    sfm_ood_per_gas = compute_per_gas_metrics(
        y_true_class=test_labels,
        y_pred_class=sfm_pred_cls,
        y_true_conc=test_concs * conc_scales,
        y_pred_conc=sfm_pred_conc * conc_scales,
        gas_names=gas_names,
    )
    sfm_ood = {
        "per_gas": sfm_ood_per_gas,
        "aggregate": compute_aggregate_metrics(sfm_ood_per_gas),
    }

    # 4. Baseline on OOD
    base_model = load_baseline(baseline_path)
    base_pred_cls, base_pred_conc = predict_plsr(base_model, ood_spectra)

    base_ood_per_gas = compute_per_gas_metrics(
        y_true_class=test_labels,
        y_pred_class=base_pred_cls,
        y_true_conc=test_concs * conc_scales,
        y_pred_conc=base_pred_conc * conc_scales,
        gas_names=gas_names,
    )
    base_ood = {
        "per_gas": base_ood_per_gas,
        "aggregate": compute_aggregate_metrics(base_ood_per_gas),
    }

    # 5. Robustness Delta Calculation
    sfm_delta = {
        "delta_r2": float(sfm_ood["aggregate"].get("mean_r2", 0.0) - sfm_std["aggregate"].get("mean_r2", 0.0)),
        "delta_mae": float(sfm_ood["aggregate"].get("mean_mae", 0.0) - sfm_std["aggregate"].get("mean_mae", 0.0)),
        "delta_f1": float(sfm_ood["aggregate"].get("mean_f1", 0.0) - sfm_std["aggregate"].get("mean_f1", 0.0)),
    }
    base_delta = {
        "delta_r2": float(base_ood["aggregate"].get("mean_r2", 0.0) - base_std["aggregate"].get("mean_r2", 0.0)),
        "delta_mae": float(base_ood["aggregate"].get("mean_mae", 0.0) - base_std["aggregate"].get("mean_mae", 0.0)),
        "delta_f1": float(base_ood["aggregate"].get("mean_f1", 0.0) - base_std["aggregate"].get("mean_f1", 0.0)),
    }

    return {
        "noise_factor": noise_factor,
        "sfm_standard": sfm_std,
        "sfm_ood": sfm_ood,
        "sfm_delta": sfm_delta,
        "baseline_standard": base_std,
        "baseline_ood": base_ood,
        "baseline_delta": base_delta,
    }


def run_full_evaluation(
    sfm_checkpoint: str,
    baseline_path: str,
    data_dir: str,
    ood_noise_factor: float,
    output_dir: str,
    device_pref: str = "auto",
    seed: int = 42,
) -> None:
    """Run end-to-end benchmark and robustness evaluation, saving JSON reports and printing summaries.

    Args:
        sfm_checkpoint: Path to fine-tuned downstream SFM checkpoint.
        baseline_path: Path to serialized baseline model pickle.
        data_dir: Directory containing processed spectral data.
        ood_noise_factor: OOD noise multiplier factor.
        output_dir: Directory where evaluation JSON artifacts will be written.
        device_pref: Compute device preference.
        seed: Random seed.
    """
    set_seed(seed)
    device = get_device(device_pref)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Executing comprehensive benchmark: SFM vs. Baseline...")
    robustness_results = evaluate_robustness(
        sfm_checkpoint=sfm_checkpoint,
        baseline_path=baseline_path,
        data_dir=data_dir,
        noise_factor=ood_noise_factor,
        device=device,
        seed=seed,
    )

    benchmark_results = {
        "sfm": robustness_results["sfm_standard"],
        "baseline": robustness_results["baseline_standard"],
    }

    # Save JSON artifacts
    bench_file = os.path.join(output_dir, "benchmark_results.json")
    robust_file = os.path.join(output_dir, "robustness_results.json")

    with open(bench_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)

    with open(robust_file, "w", encoding="utf-8") as f:
        json.dump(robustness_results, f, indent=2)

    logger.info("Saved benchmark results to: %s", bench_file)
    logger.info("Saved robustness results to: %s", robust_file)

    # Print Comparative Summary Table
    sfm_agg = benchmark_results["sfm"]["aggregate"]
    base_agg = benchmark_results["baseline"]["aggregate"]
    sfm_ood_agg = robustness_results["sfm_ood"]["aggregate"]
    base_ood_agg = robustness_results["baseline_ood"]["aggregate"]

    print("\n" + "=" * 80)
    print("SPECTRAL FOUNDATION MODEL vs. CLASSICAL BASELINE BENCHMARK")
    print("=" * 80)
    header_str = (
        f"{'Metric':<20} | {'SFM (Standard)':<15} | {'Baseline (Standard)':<18} | "
        f"{'SFM (OOD)':<12} | {'Baseline (OOD)':<14}"
    )
    print(header_str)
    print("-" * 80)
    r2_str = (
        f"{'Mean R²':<20} | {sfm_agg.get('mean_r2', 0.0):<15.4f} | {base_agg.get('mean_r2', 0.0):<18.4f} | "
        f"{sfm_ood_agg.get('mean_r2', 0.0):<12.4f} | {base_ood_agg.get('mean_r2', 0.0):<14.4f}"
    )
    print(r2_str)
    mae_str = (
        f"{'Mean MAE':<20} | {sfm_agg.get('mean_mae', 0.0):<15.4f} | {base_agg.get('mean_mae', 0.0):<18.4f} | "
        f"{sfm_ood_agg.get('mean_mae', 0.0):<12.4f} | {base_ood_agg.get('mean_mae', 0.0):<14.4f}"
    )
    print(mae_str)
    f1_str = (
        f"{'Mean F1 Score':<20} | {sfm_agg.get('mean_f1', 0.0):<15.4f} | {base_agg.get('mean_f1', 0.0):<18.4f} | "
        f"{sfm_ood_agg.get('mean_f1', 0.0):<12.4f} | {base_ood_agg.get('mean_f1', 0.0):<14.4f}"
    )
    print(f1_str)
    print("=" * 80 + "\n")
