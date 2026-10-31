"""Integration test for full benchmarking and robustness evaluation pipeline."""

import json
import os
import shutil
import numpy as np

from sfm.evaluation.baseline import train_and_save_baseline
from sfm.evaluation.evaluate import generate_ood_test_data, run_full_evaluation
from sfm.models.mae import SpectralMAE
from sfm.training.finetune import finetune
from sfm.training.utils import save_checkpoint


def test_generate_ood_test_data(tmp_path):
    """Verify OOD data generation applies perturbations correctly."""
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)
    fixtures_dir = os.path.join(os.path.dirname(__file__), "..", "fixtures")
    shutil.copy(os.path.join(fixtures_dir, "mini_labels.npz"), os.path.join(data_dir, "mini_labels.npz"))
    shutil.copy(os.path.join(fixtures_dir, "wavenumbers.npy"), os.path.join(data_dir, "wavenumbers.npy"))

    ood_spectra, labels, concs = generate_ood_test_data(data_dir, noise_factor=2.0, seed=42)
    assert ood_spectra.ndim == 2
    assert ood_spectra.shape[1] == 200
    assert labels.ndim == 2
    assert concs.ndim == 2
    assert np.all(ood_spectra >= 0.0)


def test_full_evaluation_mini(tmp_path):
    """Verify end-to-end benchmarking between SFM and baseline produces complete JSON artifacts."""
    data_dir = str(tmp_path / "data")
    checkpoints_dir = str(tmp_path / "checkpoints")
    results_dir = str(tmp_path / "results")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    fixtures_dir = os.path.join(os.path.dirname(__file__), "..", "fixtures")
    shutil.copy(os.path.join(fixtures_dir, "mini_labels.npz"), os.path.join(data_dir, "mini_labels.npz"))
    shutil.copy(os.path.join(fixtures_dir, "wavenumbers.npy"), os.path.join(data_dir, "wavenumbers.npy"))

    # 1. Train mini pretrain & downstream checkpoint
    mae = SpectralMAE(
        num_channels=200,
        patch_size=10,
        encoder_dim=32,
        encoder_depth=2,
        encoder_heads=2,
        decoder_dim=16,
        decoder_depth=1,
        decoder_heads=2,
        mask_ratio=0.75,
    )
    pretrain_ckpt = os.path.join(checkpoints_dir, "pretrain.pt")
    save_checkpoint(
        model=mae,
        optimizer=None,
        config={
            "model": {
                "num_channels": 200,
                "patch_size": 10,
                "encoder_dim": 32,
                "encoder_layers": 2,
                "encoder_heads": 2,
                "decoder_dim": 16,
                "decoder_layers": 1,
                "decoder_heads": 2,
                "mask_ratio": 0.75,
            }
        },
        epoch=1,
        loss=0.5,
        seed=42,
        path=pretrain_ckpt,
    )

    finetune_config = {
        "head": {"hidden_dim": 16, "dropout": 0.0},
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 1e-2,
            "weight_decay": 0.01,
            "freeze_backbone": True,
        },
        "data": {"num_workers": 0},
    }
    sfm_checkpoint = finetune(
        config=finetune_config,
        pretrained_checkpoint=pretrain_ckpt,
        data_dir=data_dir,
        output_dir=checkpoints_dir,
        label_fraction=1.0,
        seed=42,
        device_pref="cpu",
    )

    # 2. Train mini baseline model
    baseline_path = os.path.join(results_dir, "baseline_plsr.pkl")
    train_and_save_baseline(
        data_dir=data_dir,
        method="plsr",
        n_components=2,
        output_path=baseline_path,
        seed=42,
    )

    # 3. Run full evaluation
    run_full_evaluation(
        sfm_checkpoint=sfm_checkpoint,
        baseline_path=baseline_path,
        data_dir=data_dir,
        ood_noise_factor=2.0,
        output_dir=results_dir,
        device_pref="cpu",
        seed=42,
    )

    bench_path = os.path.join(results_dir, "benchmark_results.json")
    robust_path = os.path.join(results_dir, "robustness_results.json")

    assert os.path.exists(bench_path)
    assert os.path.exists(robust_path)

    with open(bench_path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    assert "sfm" in bench_data
    assert "baseline" in bench_data
    assert "aggregate" in bench_data["sfm"]
    assert "aggregate" in bench_data["baseline"]

    with open(robust_path, "r", encoding="utf-8") as f:
        robust_data = json.load(f)
    assert "sfm_delta" in robust_data
    assert "baseline_delta" in robust_data
    assert "delta_r2" in robust_data["sfm_delta"]
    assert "delta_r2" in robust_data["baseline_delta"]
