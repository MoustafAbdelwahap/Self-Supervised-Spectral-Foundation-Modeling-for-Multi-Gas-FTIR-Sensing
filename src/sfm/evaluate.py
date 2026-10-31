"""Command-line interface for benchmarking SFM vs. Baseline and robustness evaluation.

Usage:
    python -m sfm.evaluate \
      --sfm-checkpoint checkpoints/finetune_best.pt \
      --baseline-model results/baseline_plsr.pkl \
      --data-dir data/processed \
      --ood-noise-factor 2.0 \
      --output-dir results \
      --seed 42 \
      --device auto
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from sfm.evaluation.evaluate import run_full_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sfm.evaluate")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Benchmark fine-tuned SFM against classical baselines and evaluate OOD robustness."
    )
    parser.add_argument(
        "--sfm-checkpoint",
        type=str,
        required=True,
        help="Path to fine-tuned downstream SFM checkpoint (.pt).",
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        required=True,
        help="Path to serialized baseline model pickle (.pkl).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing processed dataset files and splits.",
    )
    parser.add_argument(
        "--ood-noise-factor",
        type=float,
        default=2.0,
        help="Multiplier factor for OOD Gaussian noise perturbation (default: 2.0).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where benchmark_results.json and robustness_results.json will be saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for evaluation reproducibility (default: 42).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Compute device preference (default: auto).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for evaluation."""
    args = parse_args()

    # 1. Validate inputs
    if not os.path.exists(args.sfm_checkpoint):
        logger.error("SFM checkpoint not found: %s", args.sfm_checkpoint)
        sys.exit(1)

    if not os.path.exists(args.baseline_model):
        logger.error("Baseline model file not found: %s", args.baseline_model)
        sys.exit(1)

    if not os.path.exists(args.data_dir):
        logger.error("Data directory does not exist: %s", args.data_dir)
        sys.exit(2)

    # 2. Run full evaluation
    try:
        run_full_evaluation(
            sfm_checkpoint=args.sfm_checkpoint,
            baseline_path=args.baseline_model,
            data_dir=args.data_dir,
            ood_noise_factor=args.ood_noise_factor,
            output_dir=args.output_dir,
            device_pref=args.device,
            seed=args.seed,
        )
        logger.info("Evaluation completed successfully. Results saved in: %s", args.output_dir)
        sys.exit(0)
    except FileNotFoundError as e:
        logger.error("Model or checkpoint error: %s", e)
        sys.exit(1)
    except ValueError as e:
        logger.error("Data or split error during evaluation: %s", e)
        sys.exit(2)
    except Exception as e:
        logger.exception("Unexpected error during evaluation: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

