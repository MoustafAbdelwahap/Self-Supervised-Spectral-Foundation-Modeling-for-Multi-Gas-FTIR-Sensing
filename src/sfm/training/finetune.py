"""Downstream multi-gas identification and quantification fine-tuning pipeline.

Freezes pre-trained backbone features, attaches a dual classification/regression head,
and trains with AdamW on labeled spectra (default 10% label budget).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sfm.data.dataset import create_dataloaders
from sfm.evaluation.metrics import compute_per_gas_metrics
from sfm.models.heads import create_downstream_model
from sfm.models.losses import DownstreamLoss
from sfm.training.utils import get_device, save_checkpoint, set_seed

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

logger = logging.getLogger(__name__)


def finetune_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    pos_weight: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Train downstream head for a single epoch.

    Args:
        model: DownstreamModel instance.
        dataloader: Training DataLoader emitting (spectra, labels, concentrations).
        optimizer: PyTorch optimizer configured for head parameters.
        loss_fn: Combined DownstreamLoss function.
        device: Active compute device.
        pos_weight: Optional per-class positive weights tensor for BCEWithLogitsLoss
            to counteract class imbalance (rare gases like CO, CH4, N2O).

    Returns:
        Dictionary with mean epoch "total_loss", "class_loss", "conc_loss".
    """
    model.train()
    total_loss_accum = 0.0
    class_loss_accum = 0.0
    conc_loss_accum = 0.0
    batch_count = 0

    for spectra, labels, concentrations in dataloader:
        spectra = spectra.to(device)
        labels = labels.to(device)
        concentrations = concentrations.to(device)

        optimizer.zero_grad()
        class_logits, conc_preds = model(spectra)

        if pos_weight is not None:
            pw = pos_weight.to(device)
            loss_class = F.binary_cross_entropy_with_logits(
                class_logits, labels.float(), pos_weight=pw
            )
        else:
            loss_class = F.binary_cross_entropy_with_logits(class_logits, labels.float())
        loss_conc = F.mse_loss(conc_preds, concentrations.float())
        total_loss = loss_class + loss_conc

        total_loss.backward()
        optimizer.step()

        total_loss_accum += float(total_loss.item())
        class_loss_accum += float(loss_class.item())
        conc_loss_accum += float(loss_conc.item())
        batch_count += 1

    denom = max(1, batch_count)
    return {
        "total_loss": total_loss_accum / denom,
        "class_loss": class_loss_accum / denom,
        "conc_loss": conc_loss_accum / denom,
    }


def evaluate(
    model: nn.Module,
    dataloader: Optional[DataLoader],
    device: torch.device,
    gas_names: Optional[List[str]] = None,
    conc_scales: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Evaluate downstream model performance across classification and regression metrics.

    Args:
        model: DownstreamModel instance.
        dataloader: Validation or test DataLoader emitting (spectra, labels, concentrations).
        device: Active compute device.
        gas_names: Optional list of gas names for metric keys.
        conc_scales: Optional per-gas max scaling factors to convert normalized concentrations
            back to raw ppm units for MAE calculation. If None, auto-detected from dataset.

    Returns:
        Dictionary containing aggregate loss, per-gas metrics, and aggregate metrics.
    """
    if dataloader is None or len(dataloader.dataset) == 0:
        return {"loss": 0.0, "per_gas": {}, "aggregate": {}}

    model.eval()
    all_class_logits = []
    all_class_targets = []
    all_conc_preds = []
    all_conc_targets = []
    total_loss_accum = 0.0
    batch_count = 0

    with torch.no_grad():
        for spectra, labels, concentrations in dataloader:
            spectra = spectra.to(device)
            labels = labels.to(device)
            concentrations = concentrations.to(device)

            class_logits, conc_preds = model(spectra)

            loss_class = F.binary_cross_entropy_with_logits(class_logits, labels.float())
            loss_conc = F.mse_loss(conc_preds, concentrations.float())
            batch_loss = loss_class + loss_conc

            total_loss_accum += float(batch_loss.item())
            batch_count += 1

            all_class_logits.append(class_logits.cpu())
            all_class_targets.append(labels.cpu())
            all_conc_preds.append(conc_preds.cpu())
            all_conc_targets.append(concentrations.cpu())

    if not all_class_logits:
        return {"loss": 0.0, "per_gas": {}, "aggregate": {}}

    cat_class_logits = torch.cat(all_class_logits, dim=0)
    cat_class_targets = torch.cat(all_class_targets, dim=0)
    cat_conc_preds = torch.cat(all_conc_preds, dim=0)
    cat_conc_targets = torch.cat(all_conc_targets, dim=0)

    y_true_cls = cat_class_targets.numpy()
    y_pred_cls = (torch.sigmoid(cat_class_logits).numpy() >= 0.5).astype(int)
    y_true_reg = cat_conc_targets.numpy()
    y_pred_reg = cat_conc_preds.numpy()

    # De-normalize concentrations back to ppm for physical MAE interpretation
    if conc_scales is None and dataloader is not None and hasattr(dataloader, "dataset"):
        ds = dataloader.dataset
        while hasattr(ds, "dataset"):
            ds = ds.dataset
        conc_scales = getattr(ds, "conc_scales", None)

    if conc_scales is not None:
        y_true_reg = y_true_reg * conc_scales
        y_pred_reg = y_pred_reg * conc_scales

    per_gas_metrics = compute_per_gas_metrics(
        y_true_class=y_true_cls,
        y_pred_class=y_pred_cls,
        y_true_conc=y_true_reg,
        y_pred_conc=y_pred_reg,
        gas_names=gas_names if gas_names else [f"Gas_{i}" for i in range(y_true_cls.shape[1])],
    )
    from sfm.evaluation.metrics import compute_aggregate_metrics
    agg_metrics = compute_aggregate_metrics(per_gas_metrics)

    return {
        "loss": total_loss_accum / max(1, batch_count),
        "per_gas": per_gas_metrics,
        "aggregate": agg_metrics,
    }


def finetune(
    config: Dict[str, Any],
    pretrained_checkpoint: str,
    data_dir: str,
    output_dir: str,
    label_fraction: float = 0.1,
    seed: int = 42,
    device_pref: str = "auto",
) -> str:
    """Run full downstream fine-tuning routine on labeled spectra.

    Args:
        config: Loaded finetune YAML configuration.
        pretrained_checkpoint: Filepath to pre-trained SpectralMAE weights.
        data_dir: Directory containing processed data and gas metadata.
        output_dir: Directory to save fine-tuned checkpoints and training logs.
        label_fraction: Fraction of labeled training data to use (default 0.1 = 10%).
        seed: Random seed for reproducibility.
        device_pref: Compute device preference ('cpu', 'cuda', or 'auto').

    Returns:
        Filepath to the saved best downstream checkpoint.
    """
    set_seed(seed)
    device = get_device(device_pref)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Discover target gases and count
    gas_metadata_path = os.path.join(data_dir, "gas_metadata.json")
    gas_names: List[str] = []
    if os.path.exists(gas_metadata_path):
        try:
            with open(gas_metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                if isinstance(meta, list):
                    # Support both "name" and legacy "gas" keys
                    gas_names = []
                    for item in meta:
                        if isinstance(item, dict):
                            gas_names.append(item.get("name", item.get("gas", str(item))))
                        else:
                            gas_names.append(str(item))
                elif isinstance(meta, dict):
                    gas_names = meta.get("gas_names", list(meta.keys()))
        except Exception as e:
            logger.warning("Could not parse gas_metadata.json: %s", e)

    # Fallback to inspecting labeled dataset if metadata is missing
    if not gas_names:
        for fname in ("labeled_spectra.npz", "mini_labels.npz"):
            npz_path = os.path.join(data_dir, fname)
            if os.path.exists(npz_path):
                data = np.load(npz_path)
                if "gas_names" in data:
                    gas_names = list(data["gas_names"])
                elif "gas_labels" in data:
                    num_found = data["gas_labels"].shape[1]
                    gas_names = [f"Gas_{i}" for i in range(num_found)]
                break

    if not gas_names:
        gas_names = ["Gas_0", "Gas_1", "Gas_2"]
    num_gases = len(gas_names)
    logger.info("Target gases for fine-tuning (%d): %s", num_gases, gas_names)

    # 2. Build downstream model
    head_cfg = config.get("head", {})
    freeze_backbone = config.get("training", {}).get("freeze_backbone", True)
    model = create_downstream_model(
        pretrained_checkpoint_path=pretrained_checkpoint,
        head_config=head_cfg,
        num_gases=num_gases,
        device=device,
        freeze_backbone=freeze_backbone,
    )

    # 3. Create DataLoaders
    training_cfg = config.get("training", {})
    batch_size = training_cfg.get("batch_size", 128)
    num_workers = config.get("data", {}).get("num_workers", 0)

    loaders = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        label_fraction=label_fraction,
        seed=seed,
    )
    train_loader = loaders.get("train")
    val_loader = loaders.get("val")

    if train_loader is None or len(train_loader.dataset) == 0:
        raise ValueError(f"No labeled training samples found in {data_dir}")

    logger.info(
        "Fine-tuning with label_fraction=%.2f: %d train samples, %d val samples",
        label_fraction,
        len(train_loader.dataset),
        len(val_loader.dataset) if val_loader else 0,
    )

    # 4. Setup Optimizer & Scheduler
    lr = float(training_cfg.get("learning_rate", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 0.01))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    epochs = training_cfg.get("epochs", 30)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_fn = DownstreamLoss()

    # Model architecture metadata for downstream checkpoint persistence
    full_ckpt_config = dict(config)
    full_ckpt_config["model"] = {
        "num_channels": getattr(model.patch_embedding, "num_channels", 200),
        "patch_size": getattr(model.patch_embedding, "patch_size", 10),
        "encoder_dim": getattr(model.encoder, "dim", 256),
        "encoder_layers": getattr(model.encoder, "depth", 6),
        "encoder_heads": getattr(model.encoder, "num_heads", 8),
    }

    # 5. TensorBoard Logger
    log_dir = os.path.join("runs", "finetune")
    writer = SummaryWriter(log_dir=log_dir) if SummaryWriter is not None else None

    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join(output_dir, "finetune_best.pt")

    logger.info("Starting fine-tuning for %d epochs...", epochs)

    # Compute positive class weights from training set to handle severe label imbalance.
    # For rare gases (CO, CH4, N2O ~4% positive rate), this boosts recall substantially.
    pos_weight: Optional[torch.Tensor] = None
    try:
        all_train_labels: List[torch.Tensor] = []
        for _, labels_b, _ in train_loader:
            all_train_labels.append(labels_b)
        stacked = torch.cat(all_train_labels, dim=0).float()  # (N, num_gases)
        n_pos = stacked.sum(dim=0).clamp(min=1.0)
        n_neg = (stacked.shape[0] - stacked.sum(dim=0)).clamp(min=1.0)
        pos_weight = (n_neg / n_pos).clamp(max=20.0)  # Cap at 20x to prevent instability
        logger.info("Computed pos_weight for class-balanced BCE: %s", pos_weight.tolist())
    except Exception as e:
        logger.warning("Could not compute pos_weight: %s. Using unweighted BCE.", e)

    for epoch in range(1, epochs + 1):
        train_metrics = finetune_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            pos_weight=pos_weight,
        )
        scheduler.step()

        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            gas_names=gas_names,
        )
        val_loss = val_metrics.get("loss", train_metrics["total_loss"])

        # TensorBoard logging
        if writer is not None:
            writer.add_scalar("Loss/train_total", train_metrics["total_loss"], epoch)
            writer.add_scalar("Loss/train_class", train_metrics["class_loss"], epoch)
            writer.add_scalar("Loss/train_conc", train_metrics["conc_loss"], epoch)
            writer.add_scalar("Loss/val_total", val_loss, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
            if "aggregate" in val_metrics:
                writer.add_scalar("Metrics/val_mean_r2", val_metrics["aggregate"].get("mean_r2", 0.0), epoch)
                writer.add_scalar("Metrics/val_mean_f1", val_metrics["aggregate"].get("mean_f1", 0.0), epoch)

        logger.info(
            "Epoch [%d/%d] Train Loss: %.4f (Cls: %.4f, Conc: %.4f) | Val Loss: %.4f",
            epoch,
            epochs,
            train_metrics["total_loss"],
            train_metrics["class_loss"],
            train_metrics["conc_loss"],
            val_loss,
        )

        # Track and save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                config=full_ckpt_config,
                epoch=epoch,
                loss=val_loss,
                seed=seed,
                path=best_checkpoint_path,
            )

    if writer is not None:
        writer.close()

    # If no val improved (e.g. single epoch), ensure best checkpoint exists
    if not os.path.exists(best_checkpoint_path):
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            config=full_ckpt_config,
            epoch=epochs,
            loss=best_val_loss if best_val_loss < float("inf") else 0.0,
            seed=seed,
            path=best_checkpoint_path,
        )

    logger.info("Fine-tuning completed. Best checkpoint saved to %s", best_checkpoint_path)
    return best_checkpoint_path
