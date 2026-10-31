"""CLI entry point for data preparation: python -m sfm.data.prepare or python -m sfm.data."""

import argparse
import logging
import sys
from sfm.data.prepare import prepare_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Parse command line arguments and execute data preparation."""
    parser = argparse.ArgumentParser(description="Ingest and prepare FTIR spectra datasets.")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to raw dataset directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save processed files")
    parser.add_argument(
        "--format",
        type=str,
        default=None,
        choices=["csv", "npz", "hdf5"],
        help="Explicit data format (auto-detected if omitted)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for dataset preparation",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum number of spectra to ingest",
    )

    args = parser.parse_args()

    try:
        mmap_p, labeled_p, wn_p, meta_p = prepare_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
        )
        print("Data preparation finished successfully:")
        print(f"  - Pre-training memmap : {mmap_p}")
        print(f"  - Labeled dataset     : {labeled_p}")
        print(f"  - Wavenumbers array   : {wn_p}")
        print(f"  - Gas metadata        : {meta_p}")
        sys.exit(0)
    except ValueError as e:
        logger.error("Validation error during data preparation: %s", e)
        sys.exit(1)
    except (IOError, FileNotFoundError) as e:
        logger.error("I/O error during data preparation: %s", e)
        sys.exit(2)
    except Exception as e:
        logger.exception("Unexpected failure: %s", e)
        sys.exit(3)


if __name__ == "__main__":
    main()

