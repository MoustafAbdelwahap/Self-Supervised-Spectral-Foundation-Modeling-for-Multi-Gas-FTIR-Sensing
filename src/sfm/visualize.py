"""CLI entry point for latent embedding visualization: python -m sfm.visualize."""

import argparse
import logging
import sys
from sfm.visualization.embeddings import visualize_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Parse arguments and visualize latent embeddings."""
    parser = argparse.ArgumentParser(description="Visualize latent representations from SpectralMAE.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pre-trained model checkpoint")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output", type=str, required=True, help="Path for output PNG plot")
    parser.add_argument("--method", type=str, default="umap", choices=["umap", "tsne"], help="Reduction technique")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of spectra to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    try:
        visualize_embeddings(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            output_path=args.output,
            method=args.method,
            num_samples=args.num_samples,
            seed=args.seed,
        )
        print(f"Visualization saved successfully to: {args.output}")
        sys.exit(0)
    except Exception as e:
        logger.exception("Failed to generate embedding visualization: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

