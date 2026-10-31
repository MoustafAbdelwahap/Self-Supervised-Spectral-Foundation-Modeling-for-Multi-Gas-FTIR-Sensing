"""Command-line interface for downstream fine-tuning.

Usage:
    python -m sfm.finetune \
      --checkpoint checkpoints/pretrain_best.pt \
      --data-dir data/processed \
      --config configs/finetune.yaml \
      --output-dir checkpoints \
      --label-fraction 0.1 \
      --seed 42 \
      --device auto
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from sfm.config import load_config, validate_finetune_config
from sfm.training.finetune import finetune

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sfm.finetune")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Fine-tune pre-trained Spectral Foundation Model on labeled multi-gas spectra."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to pre-trained SpectralMAE model checkpoint (.pt).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing processed spectra, labels, and gas metadata.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to fine-tuning YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where fine-tuned checkpoints and training logs will be saved.",
    )
    parser.add_argument(
        "--label-fraction",
        type=float,
        default=0.1,
        help="Fraction of labeled data to use for training (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Compute device preference (default: auto).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for training epochs count.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for downstream fine-tuning."""
    args = parse_args()

    # 1. Validate checkpoint path
    if not os.path.exists(args.checkpoint):
        logger.error("Pretrained checkpoint not found at: %s", args.checkpoint)
        sys.exit(2)

    # 2. Validate data directory
    if not os.path.exists(args.data_dir):
        logger.error("Data directory does not exist: %s", args.data_dir)
        sys.exit(2)

    # 3. Load and validate fine-tuning configuration
    try:
        config = load_config(args.config)
        if args.epochs is not None:
            config.setdefault("training", {})["epochs"] = args.epochs
        validate_finetune_config(config)
    except (ValueError, FileNotFoundError) as e:
        logger.error("Invalid configuration: %s", e)
        sys.exit(1)

    # 4. Execute fine-tuning
    try:
        best_ckpt = finetune(
            config=config,
            pretrained_checkpoint=args.checkpoint,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            label_fraction=args.label_fraction,
            seed=args.seed,
            device_pref=args.device,
        )
        logger.info("Downstream fine-tuning finished successfully! Best checkpoint: %s", best_ckpt)
        sys.exit(0)
    except ValueError as e:
        logger.error("Data or setup error during fine-tuning: %s", e)
        sys.exit(2)
    except Exception as e:
        logger.exception("Unexpected error during fine-tuning: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

