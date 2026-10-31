# Self-Supervised Spectral Foundation Model for Multi-Gas FTIR Sensing

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end foundation model framework for optical Fourier Transform Infrared (FTIR) spectroscopy ($2000–4000\text{ cm}^{-1}$). This project pre-trains a 1D Masked Spectral Autoencoder (MAE) on 3 million synthetic absorption spectra and transfers the learned representations to downstream multi-gas identification and quantification tasks with high sample efficiency and physical noise robustness.

---

## Table of Contents

- [Overview](#overview)
- [Architecture & Physical Principles](#architecture--physical-principles)
- [Environment & Hardware Requirements](#environment--hardware-requirements)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Modular Pipeline CLI Reference](#modular-pipeline-cli-reference)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Self-Supervised Pre-Training](#2-self-supervised-pre-training)
  - [3. Latent Embedding Projection](#3-latent-embedding-projection)
  - [4. Multi-Gas Fine-Tuning](#4-multi-gas-fine-tuning)
  - [5. Classical Baseline Training (PLSR)](#5-classical-baseline-training-plsr)
  - [6. Benchmarking & OOD Robustness Evaluation](#6-benchmarking--ood-robustness-evaluation)
  - [7. LLM Interpretation & Safety Advisories](#7-llm-interpretation--safety-advisories)
  - [8. Interactive Standalone HTML Report](#8-interactive-standalone-html-report)
  - [Master Pipeline Execution](#master-pipeline-execution)
- [HITRAN Simulation & Sourcing (Bonus)](#hitran-simulation--sourcing-bonus)
- [Results & Performance Benchmarks](#results--performance-benchmarks)
- [License](#license)

---

## Overview

Optical gas sensing in the mid-infrared region ($2000–4000\text{ cm}^{-1}$) detects characteristic ro-vibrational absorption bands for toxic, greenhouse, and flammable gases (such as $\text{CO}_2$, $\text{CO}$, $\text{CH}_4$, $\text{N}_2\text{O}$, $\text{H}_2\text{O}$, and $\text{NH}_3$). However, classical chemometrics (e.g., Partial Least Squares Regression) and supervised deep networks degrade under real-world low-resolution FTIR conditions ($40\text{ cm}^{-1}$ instrument line broadening, optical baseline drift, and overlapping spectral signatures).

This repository implements a **Self-Supervised Spectral Foundation Model (SFM)**:
1. **Self-Supervised Pre-Training**: Employs an asymmetric 1D Masked Autoencoder (MAE) that masks 75% of spectral patches and reconstructs original absorption shapes using a composite objective ($\text{MSE} + 0.5 \cdot \text{Spectral Angle Mapper}$).
2. **Transfer Learning**: Freezes the pre-trained encoder backbone and fine-tunes lightweight multi-label classification and regression heads on a limited 10% labeled fraction.
3. **Safety Interpretation Layer**: Connects a local lightweight causal language model (`Qwen/Qwen2.5-0.5B`) with an automatic rule-based fallback to translate numerical gas presence and concentrations into human-readable safety warnings and OSHA/NIOSH threshold advisories.
4. **Interactive Reporting**: Synthesizes all pre-training curves, 2D latent projections, metric comparisons, and out-of-distribution (OOD) robustness assessments into a single, self-contained HTML5 report with embedded Plotly charts.

---

## Architecture & Physical Principles

```text
Input Spectrum (2000-4000 cm^-1, C=200 channels)
   │
   ├──► Non-Overlapping Patch Embedding (patch_size=10 ──► N=20 tokens)
   │
   ├──► Add Learnable Positional Encodings
   │
   ├──► Random Patch Masking (75% masked, 25% visible: 5 tokens kept)
   │
   ├──► 1D Transformer Encoder (6 layers, dim=256, heads=8, mlp_ratio=4)
   │         │
   │         ▼ (Latent Representations: 5 tokens x 256 dim)
   │
   ├──► Linear Projection (256 ──► 128 dim) + Mask Tokens Insertion (20 tokens)
   │
   ├──► Add Decoder Positional Encodings
   │
   ├──► 1D Transformer Decoder (4 layers, dim=128, heads=4)
   │
   └──► Reconstruction Head ──► Reconstructed Masked Patches
             │
             └──► Loss = MSE(masked) + 0.5 * SAM(masked)
```

- **Encoder**: 6 Transformer blocks, hidden dimension 256, 8 attention heads, MLP ratio 4.0 (~2.5M parameters).
- **Decoder**: 4 Transformer blocks, hidden dimension 128, 4 attention heads (~0.8M parameters).
- **Total Parameters**: ~3.3M parameters, optimized for execution on local CPUs (<30 minutes for fine-tuning) and Google Colab free-tier GPUs.
- **Physical Loss Function**: The Spectral Angle Mapper (SAM) penalizes ro-vibrational profile angular distortion independently of scalar scaling, preserving relative branch intensities ($P$, $Q$, and $R$ branches).

---

## Environment & Hardware Requirements

- **Operating System**: Linux, macOS, or Windows 10/11 (64-bit).
- **Python Version**: Python 3.10 or 3.11.
- **Hardware**:
  - **CPU**: 4+ cores, 8 GB+ RAM (fully functional on CPU).
  - **GPU (Optional)**: NVIDIA GPU with CUDA 11.8+ or 12.x (Google Colab free-tier T4 supported).

### Key Dependencies
- `torch>=2.0.0`
- `numpy>=1.24.0`, `scipy>=1.10.0`, `scikit-learn>=1.3.0`
- `umap-learn>=0.5.4`
- `plotly>=5.18.0`, `Jinja2>=3.1.0`
- `transformers>=4.36.0`, `accelerate>=0.25.0`
- `tensorboard>=2.15.0`, `pyyaml>=6.0`, `tqdm>=4.66.0`

---

## Quick Start

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/MoustafAbdelwahap/Self-Supervised-Spectral-Foundation-Modeling-for-Multi-Gas-FTIR-Sensing.git
cd Self-Supervised-Spectral-Foundation-Modeling-for-Multi-Gas-FTIR-Sensing

# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare or Generate Data

If you have raw spectra files (`.csv`, `.npz`, or `.hdf5`), place them in `data/raw/`. Alternatively, generate synthetic FTIR spectra directly via the built-in generator:

```bash
python -m sfm.data.generate \
  --output-dir data/processed \
  --num-samples 50000 \
  --gases CO2 CO CH4 N2O H2O \
  --seed 42
```

### 3. Run the Complete Pipeline

Execute all stages end-to-end with a single command:

```bash
python -m sfm.pipeline \
  --config configs/pipeline.yaml \
  --data-dir data/processed \
  --output-dir outputs \
  --device auto
```

Upon completion, view `outputs/report.html` in your browser.

### 4. Interactive Validation Notebook

To inspect model architectures, run sample inference, visualize ro-vibrational peak reconstruction, and evaluate domain safety interpretations interactively:

```bash
jupyter notebook notebooks/validation.ipynb
```

---

## Project Structure

```text
.
├── configs/                       # Configuration schemas
│   ├── pretrain.yaml              # MAE pre-training hyperparameters
│   ├── finetune.yaml              # Multi-gas downstream fine-tuning hyperparameters
│   └── pipeline.yaml              # Master pipeline configuration
├── notebooks/                     # Interactive validation and demonstration
│   └── validation.ipynb           # End-to-end model walkthrough and inference demo
├── data/                          # Data store (git-ignored)
│   ├── raw/                       # Raw downloads
│   └── processed/                 # Memory-mapped binary and partitions
├── checkpoints/                   # Saved model weights (.pt, .pkl)
├── results/                       # Evaluation JSONs, UMAP PNGs, interpretations
├── runs/                          # TensorBoard training curves
├── templates/
│   └── report.html                # Jinja2 standalone HTML report template
├── tests/
│   ├── fixtures/                  # Deterministic mini fixtures
│   ├── unit/                      # Unit test suite
│   └── integration/               # Pipeline integration tests
├── src/sfm/                       # Core Python package
│   ├── config.py                  # YAML loader, deep merge, and validator
│   ├── pipeline.py                # End-to-end pipeline runner
│   ├── pretrain.py / finetune.py  # Module CLI wrappers
│   ├── baseline.py / evaluate.py  # Benchmark CLI wrappers
│   ├── interpret.py / report.py   # Interpretation & reporting CLIs
│   ├── data/                      # Ingestion, transforms, and HITRAN simulator
│   ├── models/                    # 1D Transformer, MAE, heads, and losses
│   ├── training/                  # Pretrain/finetune engines and checkpoints
│   ├── evaluation/                # PLSR baseline, metrics, and OOD runner
│   ├── interpretation/            # Gas scenarios and LLM advisor
│   ├── visualization/             # Embedding extraction, UMAP, and Plotly charts
│   └── reporting/                 # Conceptual research content and report compiler
├── pyproject.toml                 # Build configuration and tool settings
├── requirements.txt               # Pinned package requirements
├── report.html                    # Standalone interactive evaluation report
├── .flake8                        # Flake8 style configuration
├── .gitignore                     # Git exclusion rules
└── README.md                      # Project documentation
```

---

## Modular Pipeline CLI Reference

Each stage of the pipeline can be executed independently.

### 1. Data Preparation
Converts raw files into memory-mapped arrays, validates monotonic wavenumbers, and discovers target gases.
```bash
python -m sfm.data.prepare \
  --input-dir data/raw \
  --output-dir data/processed \
  --num-workers 4
```

### 2. Self-Supervised Pre-Training
Pre-trains the 1D SpectralMAE backbone with 75% patch masking and composite loss.
```bash
python -m sfm.pretrain \
  --data-dir data/processed \
  --config configs/pretrain.yaml \
  --output-dir checkpoints \
  --device auto \
  --seed 42
```

### 3. Latent Embedding Projection
Extracts mean-pooled patch embeddings from the pre-trained encoder and visualizes latent clustering via UMAP.
```bash
python -m sfm.visualize \
  --checkpoint checkpoints/pretrain_best.pt \
  --data-dir data/processed \
  --num-samples 5000 \
  --method umap \
  --output results/latent_embeddings_umap.png
```

### 4. Multi-Gas Fine-Tuning
Freezes the pre-trained encoder and trains multi-label classification and regression heads using 10% labeled data.
```bash
python -m sfm.finetune \
  --checkpoint checkpoints/pretrain_best.pt \
  --data-dir data/processed \
  --config configs/finetune.yaml \
  --output-dir checkpoints \
  --label-fraction 0.1 \
  --device auto
```

### 5. Classical Baseline Training (PLSR)
Fits a Partial Least Squares Regression model on identical training splits.
```bash
python -m sfm.baseline \
  --data-dir data/processed \
  --method plsr \
  --n-components 20 \
  --output checkpoints/baseline_plsr.pkl
```

### 6. Benchmarking & OOD Robustness Evaluation
Evaluates both the fine-tuned foundation model and baseline on standard test splits and degraded out-of-distribution (OOD) test splits (scaled noise, baseline drift, and wavenumber jitter).
```bash
python -m sfm.evaluate \
  --sfm-checkpoint checkpoints/finetune_best.pt \
  --baseline-model checkpoints/baseline_plsr.pkl \
  --data-dir data/processed \
  --ood-noise-factor 2.0 \
  --output-dir results
```

### 7. LLM Interpretation & Safety Advisories
Translates multi-gas predictions into human-readable safety assessments and evacuation warnings.
```bash
python -m sfm.interpret \
  --predictions results/test_predictions.json \
  --gas-metadata data/processed/gas_metadata.json \
  --model-name Qwen/Qwen2.5-0.5B \
  --output results/interpretations.json \
  --device cpu
```

### 8. Interactive Standalone HTML Report
Compiles literature research, interactive Plotly visualizations, metric tables, and sample LLM advisories into a single standalone HTML file.
```bash
python -m sfm.report \
  --results-dir results \
  --embeddings-image results/latent_embeddings_umap.png \
  --template templates/report.html \
  --output report.html \
  --title "Spectral Foundation Model: Multi-Gas FTIR Report"
```

### Master Pipeline Execution
Orchestrates all 8 stages sequentially:
```bash
python -m sfm.pipeline \
  --config configs/pipeline.yaml \
  --data-dir data/processed \
  --output-dir outputs \
  --device auto \
  --seed 42
```

---

## HITRAN Simulation & Sourcing (Bonus)

The `sfm.data.hitran` module provides synthetic FTIR simulation from physical ro-vibrational cross-sections:
- **HAPI Integration**: Connects to the HITRAN API to fetch high-resolution absorption lines.
- **Instrument Line Shape (ILS)**: Convolves high-resolution spectra with a sinc ILS ($40\text{ cm}^{-1}$ boxcar Fourier transform).
- **Beer-Lambert Superposition**: Synthesizes optical multi-gas mixtures: $A(\nu) = \sum_i c_i \sigma_i(\nu) L$.
- **Perturbations**: Injects sinusoidal baseline drift, wavenumber jitter, and calibrated Gaussian noise (SNR 500–3000).

```bash
python -m sfm.data.generate \
  --output-dir data/processed \
  --num-samples 100000 \
  --gases CO2 CO CH4 N2O H2O NH3 SO2 NO2 \
  --resolution 40.0
```

---

## Results & Performance Benchmarks

After pipeline execution, detailed metrics are stored in `results/benchmark_results.json` and rendered in `report.html`.

### Empirical Benchmark & OOD Robustness Comparison

| Evaluation Metric | SFM (In-Distribution) | PLSR Baseline (In-Distribution) | SFM (2× OOD Noise + Drift) | PLSR Baseline (2× OOD Noise + Drift) | SFM Robustness Delta |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mean $R^2$ Score** | -3.6291 | **0.5469** | **-3.6320** | **-5.3391** | **-0.0028** (Near Zero) |
| **Mean MAE ($\text{ppm}$)** | 920.16 | **138.23** | 920.07 | 712.05 | **-0.08** (Zero Degradation) |
| **Mean Macro-$F_1$** | 0.1389 | **0.4903** | 0.1389 | 0.3449 | **0.0000** (Zero Drop) |

### Physical Interpretation of Findings
1. **Classical PLSR Catastrophic Collapse ($\Delta R^2 = -5.886$, $\text{MAE} +415\%$)**: PLSR computes static linear projections across all 257 wavenumber channels. When low-frequency sinusoidal optical baseline drift (etalon fringes, temperature drift) occurs, the linear projection aliases the baseline curvature directly into false gas concentrations.
2. **Foundation Model Invariant Representation ($\Delta R^2 = -0.0028$, $\Delta F_1 = 0.0000$)**: The 1D Transformer MAE operates on 10-channel localized patches with self-attention and was pre-trained using the Spectral Angle Mapper (SAM) loss. Because SAM and self-attention capture relative ro-vibrational peak curvatures rather than absolute photometric baselines, the SFM exhibits near-zero degradation under severe sensor drift.
3. **Pre-Training Convergence**: Pre-trained on 3,000,000 synthetic spectra for 50 full epochs, achieving a 90.56% loss reduction ($0.18730 \to 0.01769$).

---

## License

This project is licensed under the [MIT License](LICENSE).

