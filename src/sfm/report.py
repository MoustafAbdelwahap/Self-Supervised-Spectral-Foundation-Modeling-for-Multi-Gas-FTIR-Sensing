"""Command-line interface for standalone HTML report generation.

Usage:
    python -m sfm.report \
      --results-dir results \
      --embeddings-image results/umap_embeddings.png \
      --template templates/report.html \
      --output report.html \
      --title "Spectral Foundation Model Report"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from sfm.reporting.report import generate_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sfm.report")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for report generation."""
    parser = argparse.ArgumentParser(
        description="Compile comprehensive, standalone interactive HTML evaluation report."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing benchmark_results.json and robustness_results.json.",
    )
    parser.add_argument(
        "--embeddings-image",
        type=str,
        default=None,
        help="Optional path to 2D UMAP/t-SNE latent embedding PNG.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="templates/report.html",
        help="Path to Jinja2 HTML report template (default: templates/report.html).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Destination path for generated self-contained HTML report.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Spectral Foundation Model Report",
        help="Header title for the generated report (default: 'Spectral Foundation Model Report').",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for report generation."""
    args = parse_args()

    # 1. Validate template
    if not os.path.exists(args.template):
        logger.error("Report template file not found: %s", args.template)
        sys.exit(1)

    # 2. Validate results directory
    if not os.path.exists(args.results_dir):
        logger.error("Results directory does not exist: %s", args.results_dir)
        sys.exit(2)

    # 3. Generate report
    try:
        out_file = generate_report(
            results_dir=args.results_dir,
            embeddings_image=args.embeddings_image,
            template_path=args.template,
            output_path=args.output,
            title=args.title,
        )
        logger.info("Report compiled successfully at: %s", out_file)
        sys.exit(0)
    except FileNotFoundError as e:
        logger.error("Template or file error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error during report generation: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

