"""CLI entry point for HITRAN data simulation and synthetic FTIR generation: python -m sfm.data.generate."""

import argparse
import logging
import sys
from sfm.data.hitran import generate_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Parse command-line arguments and run synthetic dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic FTIR absorption spectra using HITRAN parameters and physical perturbations."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save generated synthetic dataset files (.mmap, .npz, .json, .npy).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3_000_000,
        help="Total number of synthetic FTIR spectra to simulate (default: 3,000,000).",
    )
    parser.add_argument(
        "--gases",
        nargs="+",
        default=["CO2", "CO", "CH4", "N2O", "H2O"],
        help="Target gas molecular identifiers (default: CO2 CO CH4 N2O H2O).",
    )
    parser.add_argument(
        "--wavenumber-min",
        type=float,
        default=2000.0,
        help="Minimum spectral wavenumber in cm^-1 (default: 2000.0).",
    )
    parser.add_argument(
        "--wavenumber-max",
        type=float,
        default=4000.0,
        help="Maximum spectral wavenumber in cm^-1 (default: 4000.0).",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=40.0,
        help="FTIR spectrometer resolution in cm^-1 (default: 40.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible synthesis (default: 42).",
    )

    args = parser.parse_args()

    try:
        mmap_p, labeled_p, wn_p, meta_p = generate_dataset(
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            gas_names=args.gases,
            wavenumber_range=(args.wavenumber_min, args.wavenumber_max),
            resolution_cm=args.resolution,
            seed=args.seed,
        )
        print("HITRAN synthetic dataset generated successfully:")
        print(f"  - Pre-training memmap : {mmap_p}")
        print(f"  - Labeled dataset     : {labeled_p}")
        print(f"  - Wavenumbers array   : {wn_p}")
        print(f"  - Gas metadata        : {meta_p}")
        sys.exit(0)
    except Exception as exc:
        logger.error("Dataset generation failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

