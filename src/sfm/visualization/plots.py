"""Plotly interactive visualization generators for spectral foundation model evaluation reports.

Provides:
- create_loss_curve: Training convergence curves.
- create_metric_comparison_table: Formatted comparative performance tables.
- create_robustness_chart: Standard vs. OOD metric degradation grouped bar charts.
- create_per_gas_bar_chart: Per-gas performance breakdown charts.
- create_reconstruction_examples: Overlay plots comparing original vs. MAE-reconstructed spectra.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

logger = logging.getLogger(__name__)


def _clean_gas_name(name: str) -> str:
    """Clean gas names from legacy dictionary string representations or raw identifiers."""
    if name.startswith("{") and ("name" in name or "gas" in name):
        match = re.search(r"['\"]name['\"]\s*:\s*['\"]([^'\"]+)['\"]", name)
        if match:
            return match.group(1)
        match_gas = re.search(r"['\"]gas['\"]\s*:\s*['\"]([^'\"]+)['\"]", name)
        if match_gas:
            return match_gas.group(1)
    return name


def create_loss_curve(
    train_losses: List[float],
    title: str = "Pre-Training Convergence Curve",
) -> str:
    """Generate an interactive loss curve line plot.

    Args:
        train_losses: List of scalar training losses across epochs.
        title: Chart title.

    Returns:
        Plotly HTML div string (full_html=False, include_plotlyjs=False).
    """
    epochs = list(range(1, len(train_losses) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=train_losses,
            mode="lines+markers",
            name="Training Loss",
            line=dict(color="#1f77b4", width=2.5),
            marker=dict(size=6, color="#1f77b4"),
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, family="sans-serif")),
        xaxis=dict(title="Epoch", gridcolor="#e5e5e5"),
        yaxis=dict(title="MAE Loss (MSE + 0.5·SAM)", gridcolor="#e5e5e5"),
        template="plotly_white",
        margin=dict(l=60, r=40, t=50, b=50),
        height=380,
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def create_metric_comparison_table(
    sfm_metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Any],
) -> str:
    """Create an interactive comparative performance table using Plotly go.Table.

    Args:
        sfm_metrics: Metrics dictionary for the foundation model.
        baseline_metrics: Metrics dictionary for the PLSR baseline.

    Returns:
        Plotly HTML div string.
    """
    sfm_per_gas = sfm_metrics.get("per_gas", {})
    base_per_gas = baseline_metrics.get("per_gas", {})

    all_gases = sorted(list(set(sfm_per_gas.keys()) | set(base_per_gas.keys())))
    if not all_gases:
        all_gases = ["Mean (Aggregate)"]

    gas_col = []
    sfm_r2_col, base_r2_col = [], []
    sfm_mae_col, base_mae_col = [], []
    sfm_f1_col, base_f1_col = [], []

    for g in all_gases:
        gas_col.append(_clean_gas_name(g))
        s_g = sfm_per_gas.get(g, {})
        b_g = base_per_gas.get(g, {})
        sfm_r2_col.append(f"{s_g.get('r2', 0.0):.4f}")
        base_r2_col.append(f"{b_g.get('r2', 0.0):.4f}")
        sfm_mae_col.append(f"{s_g.get('mae', 0.0):.4f}")
        base_mae_col.append(f"{b_g.get('mae', 0.0):.4f}")
        sfm_f1_col.append(f"{s_g.get('f1', 0.0):.4f}")
        base_f1_col.append(f"{b_g.get('f1', 0.0):.4f}")

    # Append aggregate row
    s_agg = sfm_metrics.get("aggregate", {})
    b_agg = baseline_metrics.get("aggregate", {})
    gas_col.append("<b>OVERALL AVERAGE</b>")
    sfm_r2_col.append(f"<b>{s_agg.get('mean_r2', s_agg.get('r2', 0.0)):.4f}</b>")
    base_r2_col.append(f"<b>{b_agg.get('mean_r2', b_agg.get('r2', 0.0)):.4f}</b>")
    sfm_mae_col.append(f"<b>{s_agg.get('mean_mae', s_agg.get('mae', 0.0)):.4f}</b>")
    base_mae_col.append(f"<b>{b_agg.get('mean_mae', b_agg.get('mae', 0.0)):.4f}</b>")
    sfm_f1_col.append(f"<b>{s_agg.get('mean_f1', s_agg.get('f1', 0.0)):.4f}</b>")
    base_f1_col.append(f"<b>{b_agg.get('mean_f1', b_agg.get('f1', 0.0)):.4f}</b>")

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "<b>Target Gas</b>",
                        "<b>SFM R²</b>",
                        "<b>PLSR R²</b>",
                        "<b>SFM MAE</b>",
                        "<b>PLSR MAE</b>",
                        "<b>SFM F1</b>",
                        "<b>PLSR F1</b>",
                    ],
                    fill_color="#2c3e50",
                    font=dict(color="white", size=13),
                    align="center",
                    height=30,
                ),
                cells=dict(
                    values=[
                        gas_col,
                        sfm_r2_col,
                        base_r2_col,
                        sfm_mae_col,
                        base_mae_col,
                        sfm_f1_col,
                        base_f1_col,
                    ],
                    fill_color=[["#f9f9f9", "white"] * (len(gas_col) // 2 + 1)],
                    font=dict(color="#333333", size=12),
                    align="center",
                    height=28,
                ),
            )
        ]
    )
    fig.update_layout(
        title=dict(text="Benchmark Performance Summary: SFM vs. PLSR", font=dict(size=18)),
        margin=dict(l=20, r=20, t=50, b=20),
        height=320,
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def create_robustness_chart(
    standard_metrics: Dict[str, Any],
    ood_metrics: Dict[str, Any],
    model_name: str,
) -> str:
    """Create a grouped bar chart displaying metric resilience under OOD perturbations.

    Args:
        standard_metrics: Standard in-distribution test metrics.
        ood_metrics: Out-of-distribution perturbed test metrics.
        model_name: Label for chart header ('SFM' or 'PLSR Baseline').

    Returns:
        Plotly HTML div string.
    """
    std_agg = standard_metrics.get("aggregate", {})
    ood_agg = ood_metrics.get("aggregate", {})

    metrics_labels = ["Quantification R²", "Mean Abs Error (MAE)", "Detection F1 Score"]
    std_values = [
        float(std_agg.get("mean_r2", std_agg.get("r2", 0.0))),
        float(std_agg.get("mean_mae", std_agg.get("mae", 0.0))),
        float(std_agg.get("mean_f1", std_agg.get("f1", 0.0))),
    ]
    ood_values = [
        float(ood_agg.get("mean_r2", ood_agg.get("r2", 0.0))),
        float(ood_agg.get("mean_mae", ood_agg.get("mae", 0.0))),
        float(ood_agg.get("mean_f1", ood_agg.get("f1", 0.0))),
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                name="Standard (In-Distribution)",
                x=metrics_labels,
                y=std_values,
                marker_color="#2ecc71",
            ),
            go.Bar(
                name="OOD Perturbation (2× Noise + Drift)",
                x=metrics_labels,
                y=ood_values,
                marker_color="#e74c3c",
            ),
        ]
    )
    fig.update_layout(
        title=dict(text=f"Robustness Degradation Profile: {model_name}", font=dict(size=18)),
        barmode="group",
        yaxis=dict(title="Score / Value", gridcolor="#e5e5e5"),
        template="plotly_white",
        margin=dict(l=60, r=40, t=50, b=50),
        height=380,
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def create_per_gas_bar_chart(
    per_gas_metrics: Dict[str, Dict[str, float]],
    metric_name: str = "r2",
    title: str = "Per-Gas Quantification Accuracy (R²)",
) -> str:
    """Create an interactive bar chart of a single metric across all monitored gases.

    Args:
        per_gas_metrics: Dictionary mapping gas names to metric sub-dicts.
        metric_name: Key within sub-dict ('r2', 'mae', or 'f1').
        title: Plot title.

    Returns:
        Plotly HTML div string.
    """
    gases = [_clean_gas_name(g) for g in sorted(list(per_gas_metrics.keys()))]
    values = [per_gas_metrics[g].get(metric_name, 0.0) for g in gases]

    fig = go.Figure(
        go.Bar(
            x=gases,
            y=values,
            marker_color="#3498db",
            text=[f"{v:.3f}" for v in values],
            textposition="auto",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis=dict(title="Monitored Gas Component"),
        yaxis=dict(title=metric_name.upper(), gridcolor="#e5e5e5"),
        template="plotly_white",
        margin=dict(l=60, r=40, t=50, b=50),
        height=360,
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def create_comparison_bar_chart(
    sfm_per_gas: Dict[str, Dict[str, float]],
    base_per_gas: Dict[str, Dict[str, float]],
    metric_name: str = "f1",
    title: str = "Per-Gas Detection F1 Score: SFM vs. PLSR",
) -> str:
    """Create a side-by-side grouped bar chart comparing SFM vs PLSR baseline.

    Args:
        sfm_per_gas: Dictionary mapping gas names to SFM metric dicts.
        base_per_gas: Dictionary mapping gas names to PLSR metric dicts.
        metric_name: Key within metric dict ('r2', 'mae', or 'f1').
        title: Chart title.

    Returns:
        Plotly HTML div string.
    """
    all_gases = sorted(list(set(sfm_per_gas.keys()) | set(base_per_gas.keys())))
    clean_gases = [_clean_gas_name(g) for g in all_gases]
    sfm_vals = [float(sfm_per_gas.get(g, {}).get(metric_name, 0.0)) for g in all_gases]
    base_vals = [float(base_per_gas.get(g, {}).get(metric_name, 0.0)) for g in all_gases]

    fig = go.Figure(
        data=[
            go.Bar(
                name="SFM (Spectral Foundation Model)",
                x=clean_gases,
                y=sfm_vals,
                marker_color="#2b6cb0",
                text=[f"{v:.3f}" for v in sfm_vals],
                textposition="auto",
            ),
            go.Bar(
                name="PLSR Baseline (20 Components)",
                x=clean_gases,
                y=base_vals,
                marker_color="#718096",
                text=[f"{v:.3f}" for v in base_vals],
                textposition="auto",
            ),
        ]
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        barmode="group",
        xaxis=dict(title="Monitored Gas Component"),
        yaxis=dict(title=metric_name.upper(), gridcolor="#e5e5e5"),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=60, r=40, t=50, b=70),
        height=380,
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def create_reconstruction_examples(
    original: np.ndarray,
    reconstructed: np.ndarray,
    wavenumbers: np.ndarray,
    n_examples: int = 3,
) -> str:
    """Create overlay line plots comparing original vs. MAE-reconstructed absorption spectra.

    Args:
        original: Array of original spectra (n_samples, n_channels).
        reconstructed: Array of reconstructed spectra (n_samples, n_channels).
        wavenumbers: 1D array of wavenumber frequency values (cm⁻¹).
        n_examples: Number of example overlays to plot (default: 3).

    Returns:
        Plotly HTML div string.
    """
    n_plot = min(n_examples, len(original))
    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i in range(n_plot):
        c = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=wavenumbers,
                y=original[i],
                mode="lines",
                name=f"Sample {i+1} Ground Truth",
                line=dict(color=c, width=1.8),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=wavenumbers,
                y=reconstructed[i],
                mode="lines",
                name=f"Sample {i+1} MAE Reconstruction",
                line=dict(color=c, dash="dash", width=2.0),
            )
        )

    fig.update_layout(
        title=dict(text="Masked Spectral Autoencoder Reconstruction Quality", font=dict(size=18)),
        xaxis=dict(title="Wavenumber (cm⁻¹)", gridcolor="#e5e5e5", autorange="reversed"),
        yaxis=dict(title="Absorbance (a.u.)", gridcolor="#e5e5e5"),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
        margin=dict(l=60, r=40, t=50, b=90),
        height=450,
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)

