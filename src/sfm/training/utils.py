"""Training utilities: seed management, device selection, and checkpoint I/O."""

import logging
import random
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed across all libraries for deterministic execution.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to %d", seed)


def get_device(preference: str = "auto") -> torch.device:
    """Select compute device based on availability and user preference.

    Args:
        preference: Desired device ("auto", "cuda", or "cpu").

    Returns:
        torch.device instance.
    """
    preference = preference.lower().strip()
    if preference in ("auto", "cuda") and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Selected compute device: %s", device)
    return device


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    config: Dict[str, Any],
    epoch: int,
    loss: float,
    seed: int,
    path: str,
) -> None:
    """Save model checkpoint with weights, optimizer state, and training metadata.

    Args:
        model: PyTorch model whose state_dict to save.
        optimizer: PyTorch optimizer or None.
        config: Configuration dictionary snapshot.
        epoch: Current training epoch index.
        loss: Loss metric value achieved at this epoch.
        seed: Random seed used for training.
        path: Filepath where checkpoint should be written.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "config": config,
        "epoch": epoch,
        "loss": float(loss),
        "seed": seed,
        "timestamp": datetime.utcnow().isoformat(),
    }
    torch.save(checkpoint, path)
    logger.info("Checkpoint saved to %s (epoch %d, loss: %.4f)", path, epoch, loss)


def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    """Load model checkpoint onto the requested device.

    Args:
        path: Filepath to the saved checkpoint.
        device: Device to map the loaded tensors to.

    Returns:
        Checkpoint dictionary containing weights, config, and metadata.
    """
    checkpoint = torch.load(path, map_location=device)
    logger.info(
        "Checkpoint loaded from %s (saved at %s, epoch %s, loss %s)",
        path,
        checkpoint.get("timestamp"),
        checkpoint.get("epoch"),
        checkpoint.get("loss"),
    )
    return checkpoint

