"""YAML configuration loading, merging, and validation for SFM pipeline."""

import copy
import os
from typing import Any, Dict
import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Read a YAML file and return the parsed dictionary.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge two configuration dictionaries.

    Values in override replace values in base. Nested dictionaries are merged recursively.

    Args:
        base: Base configuration dictionary.
        override: Overriding configuration dictionary.

    Returns:
        Merged configuration dictionary.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def validate_pretrain_config(config: Dict[str, Any]) -> None:
    """Validate that required pre-training configuration keys exist.

    Args:
        config: Pre-training configuration dictionary.

    Raises:
        ValueError: If any mandatory keys are missing.
    """
    missing = []
    if "model" not in config or not isinstance(config["model"], dict):
        missing.append("section 'model'")
    else:
        required_model_keys = [
            "encoder_dim",
            "encoder_layers",
            "encoder_heads",
            "decoder_dim",
            "decoder_layers",
            "decoder_heads",
            "patch_size",
            "mask_ratio",
        ]
        for k in required_model_keys:
            if k not in config["model"]:
                missing.append(f"model.{k}")

    if "training" not in config or not isinstance(config["training"], dict):
        missing.append("section 'training'")
    else:
        required_training_keys = ["epochs", "batch_size", "learning_rate"]
        for k in required_training_keys:
            if k not in config["training"]:
                missing.append(f"training.{k}")

    if missing:
        raise ValueError(
            f"Pre-training configuration is missing required keys: {', '.join(missing)}"
        )


def validate_finetune_config(config: Dict[str, Any]) -> None:
    """Validate that required fine-tuning configuration keys exist.

    Args:
        config: Fine-tuning configuration dictionary.

    Raises:
        ValueError: If any mandatory keys are missing.
    """
    missing = []
    if "head" not in config or not isinstance(config["head"], dict):
        missing.append("section 'head'")
    else:
        required_head_keys = ["hidden_dim", "dropout"]
        for k in required_head_keys:
            if k not in config["head"]:
                missing.append(f"head.{k}")

    if "training" not in config or not isinstance(config["training"], dict):
        missing.append("section 'training'")
    else:
        required_training_keys = ["epochs", "batch_size", "learning_rate", "freeze_backbone"]
        for k in required_training_keys:
            if k not in config["training"]:
                missing.append(f"training.{k}")

    if missing:
        raise ValueError(
            f"Fine-tuning configuration is missing required keys: {', '.join(missing)}"
        )

