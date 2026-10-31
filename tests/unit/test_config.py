"""Unit tests for sfm.config module."""

import pytest
import yaml
from sfm.config import (
    load_config,
    merge_configs,
    validate_finetune_config,
    validate_pretrain_config,
)


def test_load_config_valid(tmp_path):
    """Test loading a valid YAML file returns expected dictionary."""
    config_data = {
        "model": {"dim": 128},
        "training": {"epochs": 10},
    }
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f)

    loaded = load_config(str(config_file))
    assert loaded == config_data


def test_load_config_missing_file():
    """Test that loading a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_config("non_existent_file_path.yaml")
    assert "not found" in str(exc_info.value).lower()


def test_validate_pretrain_config_valid():
    """Test that a complete pre-training configuration passes validation without error."""
    valid_cfg = {
        "model": {
            "encoder_dim": 256,
            "encoder_layers": 6,
            "encoder_heads": 8,
            "decoder_dim": 128,
            "decoder_layers": 4,
            "decoder_heads": 4,
            "patch_size": 10,
            "mask_ratio": 0.75,
        },
        "training": {
            "epochs": 50,
            "batch_size": 256,
            "learning_rate": 1.5e-4,
        },
    }
    # Should not raise
    validate_pretrain_config(valid_cfg)


def test_validate_pretrain_config_missing_keys():
    """Test that missing mandatory keys in pre-training config raises ValueError."""
    incomplete_cfg = {
        "model": {
            "encoder_dim": 256,
            # missing layers, heads, etc.
        },
        "training": {
            "epochs": 10,
        },
    }
    with pytest.raises(ValueError) as exc_info:
        validate_pretrain_config(incomplete_cfg)
    msg = str(exc_info.value)
    assert "encoder_layers" in msg or "batch_size" in msg


def test_validate_finetune_config_valid():
    """Test that a complete fine-tuning configuration passes validation."""
    valid_cfg = {
        "head": {"hidden_dim": 128, "dropout": 0.1},
        "training": {
            "epochs": 30,
            "batch_size": 128,
            "learning_rate": 1e-3,
            "freeze_backbone": True,
        },
    }
    validate_finetune_config(valid_cfg)


def test_merge_configs():
    """Test deep-merging two configuration dictionaries."""
    base = {
        "model": {"dim": 128, "layers": 4},
        "training": {"lr": 0.01, "batch_size": 32},
    }
    override = {
        "model": {"layers": 6},
        "training": {"lr": 0.001},
        "extra": "value",
    }
    merged = merge_configs(base, override)
    assert merged["model"]["dim"] == 128
    assert merged["model"]["layers"] == 6
    assert merged["training"]["lr"] == 0.001
    assert merged["training"]["batch_size"] == 32
    assert merged["extra"] == "value"

