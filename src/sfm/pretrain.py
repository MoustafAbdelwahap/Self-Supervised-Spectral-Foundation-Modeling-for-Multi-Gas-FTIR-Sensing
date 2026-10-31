"""CLI entry point for pre-training: python -m sfm.pretrain."""

import argparse
import logging
import sys
from sfm.config import load_config, validate_pretrain_config
from sfm.training.pretrain import pretrain

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Parse arguments and start pre-training."""
    parser = argparse.ArgumentParser(description="Self-supervised pre-training for Spectral Foundation Model.")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing processed data")
    parser.add_argument("--config", type=str, required=True, help="Path to pretrain.yaml config")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"], help="Compute device")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint path to resume from")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max batches per epoch")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")

    args = parser.parse_args()

    try:
        config = load_config(args.config)
        if args.epochs is not None:
            config.setdefault("training", {})["epochs"] = args.epochs
        validate_pretrain_config(config)
    except Exception as e:
        logger.error("Configuration error: %s", e)
        sys.exit(1)

    try:
        best_ckpt = pretrain(
            config=config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            seed=args.seed,
            device_pref=args.device,
            resume_path=args.resume,
            max_steps=args.max_steps,
        )
        print(f"Pre-training successfully finished! Best model: {best_ckpt}")
        sys.exit(0)
    except Exception as e:
        logger.exception("Pre-training execution error: %s", e)
        sys.exit(2)


if __name__ == "__main__":
    main()

