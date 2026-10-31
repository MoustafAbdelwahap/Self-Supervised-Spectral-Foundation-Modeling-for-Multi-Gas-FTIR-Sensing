"""Integration test for interactive HTML report generation."""

import json
import os
from PIL import Image

from sfm.reporting.report import generate_report


def test_report_generation(tmp_path):
    """Verify end-to-end report generation produces a valid, self-contained HTML file."""
    results_dir = str(tmp_path / "results")
    output_html = str(tmp_path / "report.html")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Create mock benchmark_results.json
    mock_benchmark = {
        "sfm": {
            "loss": 0.045,
            "per_gas": {
                "CO2": {"r2": 0.992, "mae": 0.012, "f1": 0.985},
                "CH4": {"r2": 0.988, "mae": 0.015, "f1": 0.970},
            },
            "aggregate": {"mean_r2": 0.990, "mean_mae": 0.0135, "mean_f1": 0.9775},
        },
        "baseline": {
            "per_gas": {
                "CO2": {"r2": 0.961, "mae": 0.028, "f1": 0.920},
                "CH4": {"r2": 0.945, "mae": 0.035, "f1": 0.895},
            },
            "aggregate": {"mean_r2": 0.953, "mean_mae": 0.0315, "mean_f1": 0.9075},
        },
    }
    with open(os.path.join(results_dir, "benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump(mock_benchmark, f, indent=2)

    # 2. Create mock robustness_results.json
    mock_robustness = {
        "noise_factor": 2.0,
        "sfm_standard": mock_benchmark["sfm"],
        "sfm_ood": {
            "per_gas": {
                "CO2": {"r2": 0.981, "mae": 0.018, "f1": 0.965},
                "CH4": {"r2": 0.972, "mae": 0.021, "f1": 0.950},
            },
            "aggregate": {"mean_r2": 0.9765, "mean_mae": 0.0195, "mean_f1": 0.9575},
        },
        "sfm_delta": {"delta_r2": -0.0135, "delta_mae": 0.006, "delta_f1": -0.020},
        "baseline_standard": mock_benchmark["baseline"],
        "baseline_ood": {
            "per_gas": {
                "CO2": {"r2": 0.840, "mae": 0.065, "f1": 0.780},
                "CH4": {"r2": 0.812, "mae": 0.078, "f1": 0.750},
            },
            "aggregate": {"mean_r2": 0.826, "mean_mae": 0.0715, "mean_f1": 0.765},
        },
        "baseline_delta": {"delta_r2": -0.127, "delta_mae": 0.040, "delta_f1": -0.1425},
    }
    with open(os.path.join(results_dir, "robustness_results.json"), "w", encoding="utf-8") as f:
        json.dump(mock_robustness, f, indent=2)

    # 3. Create dummy PNG image
    img_path = str(tmp_path / "dummy_umap.png")
    img = Image.new("RGB", (100, 100), color=(73, 109, 137))
    img.save(img_path)

    # 4. Generate report
    template_path = os.path.join(os.path.dirname(__file__), "..", "..", "templates", "report.html")
    report_title = "Spectral Foundation Model Benchmark Report - Test Run"

    generated_file = generate_report(
        results_dir=results_dir,
        embeddings_image=img_path,
        template_path=template_path,
        output_path=output_html,
        title=report_title,
    )

    # 5. Assertions
    assert os.path.exists(generated_file)
    assert os.path.getsize(generated_file) > 1000  # Non-trivial HTML file size

    with open(generated_file, "r", encoding="utf-8") as f:
        content = f.read()

    assert "<!DOCTYPE html>" in content
    assert report_title in content
    assert "plotly" in content.lower()
    assert "data:image/png;base64" in content
    assert "Conceptual Research" in content

