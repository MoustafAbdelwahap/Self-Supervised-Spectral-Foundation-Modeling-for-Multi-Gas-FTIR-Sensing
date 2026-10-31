"""Integration test for end-to-end SpectralMAE pre-training workflow."""

import os
from sfm.training.pretrain import pretrain
from sfm.training.utils import get_device, load_checkpoint

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def test_pretrain_mini(tmp_path):
    """Run an end-to-end mini pre-training session on fixture spectra and verify checkpoint output."""
    output_dir = str(tmp_path / "checkpoints")

    mini_config = {
        "model": {
            "encoder_dim": 32,
            "encoder_layers": 2,
            "encoder_heads": 2,
            "decoder_dim": 16,
            "decoder_layers": 1,
            "decoder_heads": 2,
            "patch_size": 10,
            "mask_ratio": 0.75,
        },
        "training": {
            "epochs": 2,
            "batch_size": 10,
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "loss_alpha": 0.5,
            "num_workers": 0,
        },
    }

    best_ckpt = pretrain(
        config=mini_config,
        data_dir=FIXTURES_DIR,
        output_dir=output_dir,
        seed=42,
        device_pref="cpu",
    )

    assert os.path.exists(best_ckpt), f"Expected checkpoint file at {best_ckpt}"

    device = get_device("cpu")
    loaded = load_checkpoint(best_ckpt, device)
    assert "model_state_dict" in loaded
    assert loaded["epoch"] == 2
    assert loaded["loss"] > 0.0

