"""Master pipeline orchestrator for Self-Supervised Spectral Foundation Model.

Executes all 8 sequential stages:
1. Data Preparation
2. Self-Supervised MAE Pre-Training
3. Latent Representation Visualization
4. Multi-Gas Fine-Tuning
5. Classical Baseline Benchmark (PLSR)
6. Comparative & OOD Robustness Evaluation
7. LLM Interpretation & Safety Advisories
8. Standalone Interactive HTML Report Generation
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Optional

import numpy as np
import torch

from sfm.config import load_config
from sfm.data.prepare import prepare_dataset
from sfm.evaluation.baseline import train_and_save_baseline
from sfm.evaluation.evaluate import run_full_evaluation
from sfm.interpretation.llm import GasAdvisor
from sfm.reporting.report import generate_report
from sfm.training.finetune import finetune
from sfm.training.pretrain import pretrain
from sfm.training.utils import get_device
from sfm.visualization.embeddings import visualize_embeddings
from sfm.models.heads import create_downstream_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _generate_sample_predictions_for_interpretation(
    checkpoint_path: str,
    data_dir: str,
    output_path: str,
    device: torch.device,
    num_samples: int = 5,
) -> str:
    """Generate sample predictions using the fine-tuned model for LLM interpretation."""
    labeled_path = os.path.join(data_dir, "labeled_spectra.npz")
    metadata_path = os.path.join(data_dir, "gas_metadata.json")

    with open(metadata_path, "r", encoding="utf-8") as f:
        gas_meta = json.load(f)
    gas_names = [g["name"] for g in gas_meta]

    labeled_data = np.load(labeled_path)
    test_spectra = labeled_data["spectra"][-num_samples:]

    model = create_downstream_model(
        pretrained_checkpoint_path=checkpoint_path,
        head_config={"hidden_dim": 128, "dropout": 0.0},
        num_gases=len(gas_names),
        device=device,
    )
    model.eval()

    predictions_list = []
    with torch.no_grad():
        x = torch.from_numpy(test_spectra).float().to(device)
        logits, concs = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds_conc = concs.cpu().numpy()

    for i in range(len(test_spectra)):
        presence = (probs[i] >= 0.5).tolist()
        concentrations = [float(c) for c in np.clip(preds_conc[i], 0.0, None)]
        predictions_list.append({
            "sample_id": int(i + 1),
            "gas_names": gas_names,
            "presence": presence,
            "concentrations": concentrations,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions_list, f, indent=2)

    return output_path


def run_pipeline(
    config_path: str,
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    device_pref: str = "auto",
    seed: int = 42,
) -> None:
    """Run full end-to-end Spectral Foundation Model pipeline.

    Args:
        config_path: Path to pipeline master YAML configuration.
        data_dir: Optional path overriding data_dir in config.
        output_dir: Optional path overriding output_dir in config.
        device_pref: Target compute device preference ('cpu', 'cuda', 'auto').
        seed: Master random seed.
    """
    total_start_time = time.time()
    logger.info("=" * 70)
    logger.info("STARTING SPECTRAL FOUNDATION MODEL (SFM) MASTER PIPELINE")
    logger.info("=" * 70)

    # 1. Load pipeline configuration
    cfg = load_config(config_path)
    active_data_dir = data_dir or cfg.get("data_dir", "data/raw")
    active_output_dir = output_dir or cfg.get("output_dir", "outputs")
    device_pref = device_pref or cfg.get("device", "auto")
    seed = seed if seed is not None else cfg.get("seed", 42)

    # Setup directories
    processed_dir = os.path.join(active_output_dir, "processed")
    checkpoints_dir = os.path.join(active_output_dir, "checkpoints")
    results_dir = os.path.join(active_output_dir, "results")
    runs_dir = os.path.join(active_output_dir, "runs")
    for d in [processed_dir, checkpoints_dir, results_dir, runs_dir]:
        os.makedirs(d, exist_ok=True)

    device = get_device(device_pref)
    logger.info("Master Output Directory: %s", active_output_dir)
    logger.info("Execution Device: %s", device)
    logger.info("Random Seed: %d", seed)

    # ==========================================
    # Stage 1: Data Ingestion & Preparation
    # ==========================================
    logger.info("\n--- [Stage 1/8] Data Ingestion and Preparation ---")
    stage_start = time.time()
    mmap_check = os.path.join(active_data_dir, "pretrain_spectra.mmap")
    labeled_check = os.path.join(active_data_dir, "labeled_spectra.npz")
    if os.path.exists(mmap_check) and os.path.exists(labeled_check):
        logger.info("Found existing processed dataset in '%s'. Skipping ingestion.", active_data_dir)
        processed_dir = active_data_dir
    else:
        try:
            prepare_dataset(input_dir=active_data_dir, output_dir=processed_dir)
            logger.info("Data preparation completed in %.2fs", time.time() - stage_start)
        except Exception as e:
            logger.error("Stage 1 (Data Preparation) failed: %s", e)
            raise

    # ==========================================
    # Stage 2: Self-Supervised MAE Pre-Training
    # ==========================================
    logger.info("\n--- [Stage 2/8] Self-Supervised SpectralMAE Pre-Training ---")
    stage_start = time.time()
    pretrain_cfg_path = cfg.get("pretrain_config", "configs/pretrain.yaml")
    pretrain_cfg = load_config(pretrain_cfg_path)
    try:
        best_pretrain_ckpt = pretrain(
            config=pretrain_cfg,
            data_dir=processed_dir,
            output_dir=checkpoints_dir,
            seed=seed,
            device_pref=device_pref,
        )
        logger.info(
            "Pre-training completed in %.2fs. Best checkpoint: %s",
            time.time() - stage_start,
            best_pretrain_ckpt,
        )
    except Exception as e:
        logger.error("Stage 2 (Pre-Training) failed: %s", e)
        raise

    # ==========================================
    # Stage 3: Latent Embedding Visualization
    # ==========================================
    logger.info("\n--- [Stage 3/8] Latent Embedding Projection & Visualization ---")
    stage_start = time.time()
    umap_png_path = os.path.join(results_dir, "latent_embeddings_umap.png")
    try:
        visualize_embeddings(
            checkpoint_path=best_pretrain_ckpt,
            data_dir=processed_dir,
            output_path=umap_png_path,
            method="umap",
            num_samples=cfg.get("umap_num_samples", 5000),
            seed=seed,
        )
        logger.info("Embedding visualization generated at %s in %.2fs", umap_png_path, time.time() - stage_start)
    except Exception as e:
        logger.warning("Stage 3 (Visualization) encountered an issue: %s", e)
        umap_png_path = None

    # ==========================================
    # Stage 4: Multi-Gas Downstream Fine-Tuning
    # ==========================================
    logger.info("\n--- [Stage 4/8] Downstream Multi-Gas Fine-Tuning ---")
    stage_start = time.time()
    finetune_cfg_path = cfg.get("finetune_config", "configs/finetune.yaml")
    finetune_cfg = load_config(finetune_cfg_path)
    label_fraction = finetune_cfg.get("training", {}).get("label_fraction", 0.1)
    try:
        best_downstream_ckpt = finetune(
            config=finetune_cfg,
            pretrained_checkpoint=best_pretrain_ckpt,
            data_dir=processed_dir,
            output_dir=checkpoints_dir,
            label_fraction=label_fraction,
            seed=seed,
            device_pref=device_pref,
        )
        logger.info(
            "Downstream fine-tuning completed in %.2fs. Checkpoint: %s",
            time.time() - stage_start,
            best_downstream_ckpt,
        )
    except Exception as e:
        logger.error("Stage 4 (Fine-Tuning) failed: %s", e)
        raise

    # ==========================================
    # Stage 5: Classical Baseline Benchmark (PLSR)
    # ==========================================
    logger.info("\n--- [Stage 5/8] Classical Baseline Training ---")
    stage_start = time.time()
    baseline_model_path = os.path.join(checkpoints_dir, "baseline_plsr.pkl")
    baseline_method = cfg.get("baseline_method", "plsr")
    baseline_n_comp = cfg.get("baseline_n_components", 20)
    try:
        train_and_save_baseline(
            data_dir=processed_dir,
            method=baseline_method,
            n_components=baseline_n_comp,
            output_path=baseline_model_path,
            seed=seed,
        )
        logger.info(
            "Baseline training completed in %.2fs. Model saved: %s",
            time.time() - stage_start,
            baseline_model_path,
        )
    except Exception as e:
        logger.error("Stage 5 (Baseline) failed: %s", e)
        raise

    # ==========================================
    # Stage 6: Comparative & OOD Evaluation
    # ==========================================
    logger.info("\n--- [Stage 6/8] Comparative Evaluation & OOD Robustness Analysis ---")
    stage_start = time.time()
    ood_noise_factor = cfg.get("ood_noise_factor", 2.0)
    try:
        run_full_evaluation(
            sfm_checkpoint=best_downstream_ckpt,
            baseline_path=baseline_model_path,
            data_dir=processed_dir,
            ood_noise_factor=ood_noise_factor,
            output_dir=results_dir,
            device_pref=device_pref,
            seed=seed,
        )
        logger.info("Benchmarking completed in %.2fs.", time.time() - stage_start)
    except Exception as e:
        logger.error("Stage 6 (Evaluation) failed: %s", e)
        raise

    # ==========================================
    # Stage 7: LLM Interpretation Layer
    # ==========================================
    logger.info("\n--- [Stage 7/8] LLM Interpretation & Safety Advisory Generation ---")
    stage_start = time.time()
    predictions_json_path = os.path.join(results_dir, "test_predictions.json")
    interpretations_json_path = os.path.join(results_dir, "interpretations.json")
    try:
        _generate_sample_predictions_for_interpretation(
            checkpoint_path=best_downstream_ckpt,
            data_dir=processed_dir,
            output_path=predictions_json_path,
            device=device,
            num_samples=5,
        )
        with open(predictions_json_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)

        llm_model_name = cfg.get("llm_model_name", "Qwen/Qwen2.5-0.5B")
        advisor = GasAdvisor(model_name=llm_model_name, device="cpu")
        annotated_predictions = advisor.batch_interpret(predictions)

        with open(interpretations_json_path, "w", encoding="utf-8") as f:
            json.dump(annotated_predictions, f, indent=2)

        logger.info(
            "Generated safety advisories for %d samples in %.2fs",
            len(annotated_predictions),
            time.time() - stage_start,
        )
    except Exception as e:
        logger.warning("Stage 7 (LLM Interpretation) warning: %s. Continuing pipeline.", e)

    # ==========================================
    # Stage 8: Interactive HTML Report Generation
    # ==========================================
    logger.info("\n--- [Stage 8/8] Standalone Interactive HTML Report Generation ---")
    stage_start = time.time()
    report_path = os.path.join(active_output_dir, "report.html")
    template_path = os.path.join("templates", "report.html")
    report_title = cfg.get("report_title", "Spectral Foundation Model: Self-Supervised Multi-Gas FTIR Sensing")
    try:
        generate_report(
            results_dir=results_dir,
            embeddings_image=umap_png_path,
            template_path=template_path,
            output_path=report_path,
            title=report_title,
        )
        logger.info("Report compiled in %.2fs at: %s", time.time() - stage_start, report_path)
    except Exception as e:
        logger.error("Stage 8 (Report Generation) failed: %s", e)
        raise

    total_duration = time.time() - total_start_time
    logger.info("=" * 70)
    logger.info(
        "SFM PIPELINE EXECUTION COMPLETED SUCCESSFULLY in %.2f seconds (%.2f minutes)",
        total_duration,
        total_duration / 60.0,
    )
    logger.info("Report: %s", report_path)
    logger.info("Checkpoints: %s", checkpoints_dir)
    logger.info("Results: %s", results_dir)
    logger.info("=" * 70)


def main() -> None:
    """Parse CLI arguments and run pipeline."""
    parser = argparse.ArgumentParser(
        description="Master pipeline runner for Self-Supervised Spectral Foundation Model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline.yaml",
        help="Path to pipeline master YAML configuration (default: configs/pipeline.yaml).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to raw or processed dataset directory (overrides config if specified).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to save all output artifacts (overrides config if specified).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Compute device preference (default: auto).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args()

    try:
        run_pipeline(
            config_path=args.config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device_pref=args.device,
            seed=args.seed,
        )
        sys.exit(0)
    except Exception as exc:
        logger.error("Pipeline failed with exception: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

