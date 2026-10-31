"""Command-line interface for LLM natural language interpretation of gas predictions.

Usage:
    python -m sfm.interpret \
      --predictions results/predictions.json \
      --gas-metadata data/processed/gas_metadata.json \
      --model-name Qwen/Qwen2.5-0.5B \
      --output results/interpretations.json \
      --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from sfm.interpretation.llm import GasAdvisor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sfm.interpret")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for gas interpretation."""
    parser = argparse.ArgumentParser(
        description="Generate human-readable gas safety advisories using local LLM or rule engine."
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to JSON file containing multi-gas prediction dictionaries.",
    )
    parser.add_argument(
        "--gas-metadata",
        type=str,
        required=True,
        help="Path to gas_metadata.json file containing target gas definitions.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model ID for causal language generation (default: Qwen/Qwen2.5-0.5B).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path where annotated output JSON should be saved.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Compute device for LLM generation (default: cpu).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for interpretation."""
    args = parse_args()

    # 1. Validate inputs
    if not os.path.exists(args.predictions):
        logger.error("Predictions file not found: %s", args.predictions)
        sys.exit(1)

    if not os.path.exists(args.gas_metadata):
        logger.error("Gas metadata file not found: %s", args.gas_metadata)
        sys.exit(1)

    # 2. Load predictions
    try:
        with open(args.predictions, "r", encoding="utf-8") as f:
            predictions_data = json.load(f)
        if not isinstance(predictions_data, list):
            predictions_data = [predictions_data]
    except Exception as e:
        logger.error("Failed to load predictions JSON: %s", e)
        sys.exit(1)

    # 3. Load gas metadata
    try:
        with open(args.gas_metadata, "r", encoding="utf-8") as f:
            gas_meta = json.load(f)
        if isinstance(gas_meta, list):
            default_gas_names = [
                item.get("name", item.get("gas", str(item))) if isinstance(item, dict) else str(item)
                for item in gas_meta
            ]
        elif isinstance(gas_meta, dict):
            default_gas_names = gas_meta.get("gas_names", list(gas_meta.keys()))
        else:
            default_gas_names = []
    except Exception as e:
        logger.warning("Failed to load gas metadata: %s", e)
        default_gas_names = []

    # Inject default gas names if predictions do not include them
    for item in predictions_data:
        if "gas_names" not in item or not item["gas_names"]:
            item["gas_names"] = default_gas_names

    # 4. Initialize GasAdvisor
    advisor = GasAdvisor(model_name=args.model_name, device=args.device)

    # 5. Batch interpret
    try:
        annotated = advisor.batch_interpret(predictions_data)
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(annotated, f, indent=2)

        logger.info("Successfully saved %d interpreted predictions to %s", len(annotated), args.output)
        sys.exit(0)
    except Exception as e:
        logger.exception("Error during batch interpretation: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

