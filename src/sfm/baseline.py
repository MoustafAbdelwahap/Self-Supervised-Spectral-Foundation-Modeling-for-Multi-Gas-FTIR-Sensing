"""Command-line interface for training classical baseline models (PLSR / Linear LS).

Usage:
    python -m sfm.baseline \
      --data-dir data/processed \
      --method plsr \
      --n-components 20 \
      --output results/baseline_plsr.pkl \
      --seed 42
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from sfm.evaluation.baseline import train_and_save_baseline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sfm.baseline")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for baseline training."""
    parser = argparse.ArgumentParser(
        description="Train classical baseline model (PLSR or Linear Least Squares) on multi-gas FTIR spectra."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing processed dataset files and splits.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="plsr",
        choices=["plsr", "linear_ls"],
        help="Classical baseline algorithm (default: plsr).",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=20,
        help="Number of latent components for PLSR (default: 20).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path where serialized baseline model (.pkl) will be saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for baseline training."""
    args = parse_args()

    if not os.path.exists(args.data_dir):
        logger.error("Data directory does not exist: %s", args.data_dir)
        sys.exit(1)

    try:
        saved_path = train_and_save_baseline(
            data_dir=args.data_dir,
            method=args.method,
            n_components=args.n_components,
            output_path=args.output,
            seed=args.seed,
        )
        logger.info("Successfully trained and saved baseline model to: %s", saved_path)
        sys.exit(0)
    except ValueError as e:
        logger.error("Data error during baseline training: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error during baseline training: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

