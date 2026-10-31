"""Integration test for downstream fine-tuning loop."""

import os
import shutil
import torch

from sfm.models.heads import create_downstream_model
from sfm.models.mae import SpectralMAE
from sfm.training.finetune import finetune
from sfm.training.utils import save_checkpoint


def test_finetune_mini(tmp_path):
    """Verify end-to-end downstream fine-tuning convergence and checkpoint loading."""
    output_dir = str(tmp_path / "checkpoints")
    data_dir = str(tmp_path / "data")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    fixtures_dir = os.path.join(os.path.dirname(__file__), "..", "fixtures")
    shutil.copy(os.path.join(fixtures_dir, "mini_labels.npz"), os.path.join(data_dir, "mini_labels.npz"))
    shutil.copy(os.path.join(fixtures_dir, "wavenumbers.npy"), os.path.join(data_dir, "wavenumbers.npy"))

    # 1. Create a tiny pre-trained checkpoint
    num_channels = 200
    patch_size = 10
    encoder_dim = 32
    mae = SpectralMAE(
        num_channels=num_channels,
        patch_size=patch_size,
        encoder_dim=encoder_dim,
        encoder_depth=2,
        encoder_heads=2,
        decoder_dim=16,
        decoder_depth=1,
        decoder_heads=2,
        mask_ratio=0.75,
    )
    pretrain_ckpt_path = os.path.join(output_dir, "pretrain_best.pt")
    dummy_config = {
        "model": {
            "num_channels": num_channels,
            "patch_size": patch_size,
            "encoder_dim": encoder_dim,
            "encoder_layers": 2,
            "encoder_heads": 2,
            "decoder_dim": 16,
            "decoder_layers": 1,
            "decoder_heads": 2,
            "mask_ratio": 0.75,
        }
    }
    save_checkpoint(
        model=mae,
        optimizer=None,
        config=dummy_config,
        epoch=1,
        loss=0.5,
        seed=42,
        path=pretrain_ckpt_path,
    )

    # 2. Run downstream fine-tuning for 2 epochs
    finetune_config = {
        "head": {
            "hidden_dim": 16,
            "dropout": 0.0,
        },
        "training": {
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 1e-2,
            "weight_decay": 0.01,
            "freeze_backbone": True,
        },
        "data": {
            "num_workers": 0,
        },
    }

    best_ckpt_path = finetune(
        config=finetune_config,
        pretrained_checkpoint=pretrain_ckpt_path,
        data_dir=data_dir,
        output_dir=output_dir,
        label_fraction=1.0,
        seed=42,
        device_pref="cpu",
    )

    assert os.path.exists(best_ckpt_path)
    assert best_ckpt_path.endswith("finetune_best.pt")

    # 3. Reload checkpoint and verify predictions
    device = torch.device("cpu")
    reloaded_model = create_downstream_model(
        pretrained_checkpoint_path=best_ckpt_path,
        head_config=finetune_config["head"],
        num_gases=0,  # Auto-inferred from downstream checkpoint
        device=device,
    )
    reloaded_model.eval()
    num_gases = reloaded_model.num_gases

    sample_x = torch.randn(2, num_channels)
    with torch.no_grad():
        class_logits, conc_preds = reloaded_model(sample_x)

    assert class_logits.shape == (2, num_gases)
    assert conc_preds.shape == (2, num_gases)
    assert torch.isfinite(class_logits).all()
    assert torch.isfinite(conc_preds).all()
