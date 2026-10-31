"""Self-supervised pre-training loop for the Masked Spectral Autoencoder."""

import logging
import os
from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from sfm.data.dataset import create_dataloaders
from sfm.models.losses import MAELoss
from sfm.models.mae import SpectralMAE
from sfm.training.utils import get_device, load_checkpoint, save_checkpoint, set_seed

logger = logging.getLogger(__name__)


def create_model(config: Dict[str, Any], num_channels: int = 200) -> SpectralMAE:
    """Instantiate a SpectralMAE model from configuration parameters.

    Args:
        config: Loaded pre-training YAML dictionary.
        num_channels: Spectral length of input spectra.

    Returns:
        Configured SpectralMAE module.
    """
    model_cfg = config.get("model", {})
    model = SpectralMAE(
        num_channels=num_channels,
        patch_size=model_cfg.get("patch_size", 10),
        encoder_dim=model_cfg.get("encoder_dim", 256),
        encoder_depth=model_cfg.get("encoder_layers", 6),
        encoder_heads=model_cfg.get("encoder_heads", 8),
        decoder_dim=model_cfg.get("decoder_dim", 128),
        decoder_depth=model_cfg.get("decoder_layers", 4),
        decoder_heads=model_cfg.get("decoder_heads", 4),
        mask_ratio=model_cfg.get("mask_ratio", 0.75),
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Instantiated SpectralMAE: %d total params (%d trainable)", total_params, trainable_params)
    return model


def train_one_epoch(
    model: SpectralMAE,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: MAELoss,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    max_steps: Optional[int] = None,
) -> float:
    """Execute training over a single dataset epoch.

    Args:
        model: SpectralMAE model.
        dataloader: Training dataloader yielding spectral tensors.
        optimizer: Optimization algorithm.
        loss_fn: Composite MAE loss function.
        device: Device on which computation runs.
        scaler: Optional AMP gradient scaler for mixed precision.
        max_steps: Optional maximum number of batches to process this epoch.

    Returns:
        Average epoch loss.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Pre-training", leave=False)
    for batch in pbar:
        if max_steps is not None and num_batches >= max_steps:
            break

        if isinstance(batch, (tuple, list)):
            batch = batch[0]

        batch = batch.to(device)
        optimizer.zero_grad()

        use_amp = scaler is not None and device.type == "cuda"
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            pred_patches, target_patches, mask = model(batch)
            loss = loss_fn(pred_patches, target_patches, mask)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

    return total_loss / max(num_batches, 1)


def pretrain(
    config: Dict[str, Any],
    data_dir: str,
    output_dir: str,
    seed: int = 42,
    device_pref: str = "auto",
    resume_path: Optional[str] = None,
    max_steps: Optional[int] = None,
) -> str:
    """Execute full self-supervised pre-training loop.

    Args:
        config: Pre-training configuration.
        data_dir: Directory containing preprocessed spectral data.
        output_dir: Target directory for checkpoints and metrics.
        seed: Random seed for reproducibility.
        device_pref: Compute device preference ('auto', 'cuda', 'cpu').
        resume_path: Optional checkpoint filepath from which to resume.

    Returns:
        Path to best checkpoint file.
    """
    set_seed(seed)
    device = get_device(device_pref)
    os.makedirs(output_dir, exist_ok=True)

    # Determine spectral channel length
    wn_path = os.path.join(data_dir, "wavenumbers.npy")
    if os.path.exists(wn_path):
        num_channels = len(np.load(wn_path))
    else:
        num_channels = 200

    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 256)
    epochs = train_cfg.get("epochs", 50)
    lr = float(train_cfg.get("learning_rate", 1.5e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.05))
    loss_alpha = float(train_cfg.get("loss_alpha", 0.5))
    num_workers = train_cfg.get("num_workers", 2)
    effective_max_steps = max_steps if max_steps is not None else train_cfg.get("max_steps", None)

    loaders = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    dataloader = loaders.get("pretrain") or loaders.get("train")
    if dataloader is None:
        raise FileNotFoundError(f"Could not initialize a training dataloader from {data_dir}")

    model = create_model(config, num_channels=num_channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_fn = MAELoss(alpha=loss_alpha)

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    start_epoch = 0
    best_loss = float("inf")
    best_checkpoint_path = os.path.join(output_dir, "pretrain_best.pt")

    if resume_path and os.path.exists(resume_path):
        ckpt = load_checkpoint(resume_path, device)
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_loss = ckpt.get("loss", float("inf"))
        logger.info("Resuming training from epoch %d with prior best loss %.4f", start_epoch + 1, best_loss)
        if start_epoch > 0 and start_epoch < epochs:
            for _ in range(start_epoch):
                scheduler.step()

    # TensorBoard logging
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = os.path.join(output_dir, "..", "runs", "pretrain")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
    except Exception:
        writer = None

    logger.info("Starting pre-training for %d epochs on %s", epochs, device)
    try:
        for epoch in range(start_epoch, epochs):
            epoch_loss = train_one_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                scaler=scaler,
                max_steps=effective_max_steps,
            )
            scheduler.step()

            logger.info(
                "Epoch [%d/%d] - Loss: %.5f - LR: %.6f",
                epoch + 1,
                epochs,
                epoch_loss,
                scheduler.get_last_lr()[0],
            )
            if writer is not None:
                writer.add_scalar("Loss/pretrain", epoch_loss, epoch + 1)
                writer.add_scalar("LR/pretrain", scheduler.get_last_lr()[0], epoch + 1)

            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
                periodic_path = os.path.join(output_dir, f"pretrain_epoch_{epoch + 1}.pt")
                save_checkpoint(model, optimizer, config, epoch + 1, epoch_loss, seed, periodic_path)

            # Save best checkpoint
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                save_checkpoint(model, optimizer, config, epoch + 1, epoch_loss, seed, best_checkpoint_path)
                logger.info("New best pre-training model saved: loss=%.5f", best_loss)

    except KeyboardInterrupt:
        logger.warning("Pre-training interrupted by user. Saving emergency checkpoint...")
        emergency_path = os.path.join(output_dir, "pretrain_interrupted.pt")
        save_checkpoint(model, optimizer, config, epoch + 1, epoch_loss, seed, emergency_path)

    finally:
        if writer is not None:
            writer.close()

    return best_checkpoint_path
