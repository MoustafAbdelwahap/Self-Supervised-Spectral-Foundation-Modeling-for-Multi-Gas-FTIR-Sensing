"""End-to-end integration test for the master pipeline orchestrator."""

import json
import os
import numpy as np
import pytest
import yaml

from sfm.pipeline import run_pipeline


@pytest.mark.slow
def test_mini_pipeline(tmp_path):
    """Verify that run_pipeline executes all 8 sequential stages end-to-end.

    Creates lightweight configs and small fixture data, runs the pipeline on CPU,
    and asserts that all expected artifact files are created and valid.
    """
    # 1. Setup fixture data directory
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 100 spectra, 200 channels
    n_samples = 100
    n_channels = 200
    spectra = np.random.RandomState(42).uniform(0.0, 1.0, size=(n_samples, n_channels)).astype(np.float32)

    # Create memmap
    mmap_path = str(data_dir / "pretrain_spectra.mmap")
    mm = np.memmap(mmap_path, dtype="float32", mode="w+", shape=(n_samples, n_channels))
    mm[:] = spectra
    mm.flush()
    del mm

    # Wavenumbers
    wavenumbers = np.linspace(2000.0, 4000.0, n_channels, dtype=np.float32)
    np.save(str(data_dir / "wavenumbers.npy"), wavenumbers)

    # Gas metadata
    gas_names = ["CO2", "CO", "CH4"]
    metadata = [{"name": g, "index": i, "min_conc": 0.0, "max_conc": 1.0} for i, g in enumerate(gas_names)]
    with open(str(data_dir / "gas_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Labeled dataset (20 samples)
    n_labeled = 20
    labels = np.random.RandomState(42).choice([0, 1], size=(n_labeled, len(gas_names))).astype(np.int32)
    concs = np.random.RandomState(42).uniform(0.0, 1.0, size=(n_labeled, len(gas_names))).astype(np.float32) * labels
    np.savez(
        str(data_dir / "labeled_spectra.npz"),
        spectra=spectra[:n_labeled],
        gas_labels=labels,
        concentrations=concs,
        gas_names=np.array(gas_names),
    )

    # 2. Create lightweight YAML configs
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    pretrain_cfg_path = cfg_dir / "pretrain_mini.yaml"
    pretrain_cfg = {
        "model": {
            "encoder_dim": 32,
            "encoder_layers": 2,
            "encoder_heads": 2,
            "encoder_mlp_ratio": 2,
            "decoder_dim": 16,
            "decoder_layers": 1,
            "decoder_heads": 2,
            "patch_size": 10,
            "mask_ratio": 0.75,
        },
        "training": {
            "epochs": 1,
            "batch_size": 20,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "warmup_epochs": 0,
            "loss_alpha": 0.5,
            "num_workers": 0,
            "seed": 42,
        },
    }
    with open(pretrain_cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(pretrain_cfg, f)

    finetune_cfg_path = cfg_dir / "finetune_mini.yaml"
    finetune_cfg = {
        "head": {
            "hidden_dim": 16,
            "dropout": 0.0,
        },
        "training": {
            "epochs": 1,
            "batch_size": 10,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "freeze_backbone": True,
            "label_fraction": 1.0,
            "seed": 42,
        },
    }
    with open(finetune_cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(finetune_cfg, f)

    output_dir = tmp_path / "outputs"
    pipeline_cfg_path = cfg_dir / "pipeline_mini.yaml"
    pipeline_cfg = {
        "pretrain_config": str(pretrain_cfg_path),
        "finetune_config": str(finetune_cfg_path),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "device": "cpu",
        "seed": 42,
        "baseline_method": "plsr",
        "baseline_n_components": 2,
        "ood_noise_factor": 1.5,
        "umap_num_samples": 40,
        "llm_model_name": "mock-nonexistent-llm",  # Will gracefully trigger rule-based advisory fallback
        "report_title": "Mini Pipeline Integration Report",
    }
    with open(pipeline_cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(pipeline_cfg, f)

    # 3. Execute master pipeline
    run_pipeline(
        config_path=str(pipeline_cfg_path),
        device_pref="cpu",
        seed=42,
    )

    # 4. Verify all output artifacts exist
    assert os.path.exists(str(output_dir / "checkpoints" / "pretrain_best.pt"))
    assert os.path.exists(str(output_dir / "checkpoints" / "finetune_best.pt"))
    assert os.path.exists(str(output_dir / "checkpoints" / "baseline_plsr.pkl"))
    assert os.path.exists(str(output_dir / "results" / "benchmark_results.json"))
    assert os.path.exists(str(output_dir / "results" / "robustness_results.json"))
    assert os.path.exists(str(output_dir / "report.html"))

    # Verify report is populated
    report_size = os.path.getsize(str(output_dir / "report.html"))
    assert report_size > 100_000  # Standalone HTML with inline Plotly should be >100KB
    with open(str(output_dir / "report.html"), "r", encoding="utf-8") as f:
        content = f.read()
    assert "Mini Pipeline Integration Report" in content
    assert "plotly" in content.lower()

