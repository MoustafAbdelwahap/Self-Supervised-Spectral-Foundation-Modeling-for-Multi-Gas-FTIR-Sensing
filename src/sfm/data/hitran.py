"""HITRAN data sourcing, synthetic spectral generation, and FTIR instrument simulation."""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.signal
from tqdm import tqdm

logger = logging.getLogger(__name__)


# Known vibrational band centers (cm^-1) and approximate peak widths for offline simulation
SYNTHETIC_BAND_CENTERS: Dict[str, List[Tuple[float, float, float]]] = {
    # gas: list of (center_cm-1, width_cm-1, relative_intensity)
    "CO2": [(2349.0, 35.0, 1.0), (3714.0, 25.0, 0.15), (3609.0, 20.0, 0.12)],
    "CO": [(2143.0, 45.0, 0.8)],
    "CH4": [(3018.0, 50.0, 0.9), (2850.0, 30.0, 0.3)],
    "N2O": [(2224.0, 30.0, 0.85), (3480.0, 25.0, 0.1)],
    "H2O": [(3756.0, 80.0, 0.7), (3657.0, 60.0, 0.5), (2050.0, 40.0, 0.08)],
    "NH3": [(3336.0, 40.0, 0.65), (2500.0, 30.0, 0.05)],
    "SO2": [(2500.0, 25.0, 0.4)],
    "NO2": [(2906.0, 35.0, 0.5)],
}


def _generate_offline_crosssection(
    gas_name: str,
    wavenumbers: np.ndarray,
) -> np.ndarray:
    """Generate synthetic physical absorption cross-section using Lorentzian-Gaussian profiles.

    Args:
        gas_name: Gas identifier.
        wavenumbers: Array of wavenumbers (cm^-1).

    Returns:
        Array of synthetic cross-section values.
    """
    bands = SYNTHETIC_BAND_CENTERS.get(
        gas_name,
        [(float(np.mean(wavenumbers)), 40.0, 0.5)],
    )
    sigma = np.zeros_like(wavenumbers, dtype=np.float32)
    for center, width, intensity in bands:
        # Pseudo-Voigt profile approximating vibrational-rotational band envelope
        dx = (wavenumbers - center) / (width / 2.0)
        lorentz = 1.0 / (1.0 + dx**2)
        gauss = np.exp(-np.log(2.0) * dx**2)
        profile = 0.5 * lorentz + 0.5 * gauss
        sigma += intensity * profile.astype(np.float32)
    return sigma


def fetch_hitran_crosssections(
    gas_name: str,
    wavenumber_min: float,
    wavenumber_max: float,
    temperature: float = 296.0,
    pressure: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fetch or compute HITRAN absorption cross-sections using the HAPI library.

    Args:
        gas_name: Chemical formula or identifier (e.g., 'CO2', 'CH4').
        wavenumber_min: Starting wavenumber in cm^-1.
        wavenumber_max: Ending wavenumber in cm^-1.
        temperature: Temperature in Kelvin (default 296.0 K).
        pressure: Pressure in atmospheres (default 1.0 atm).

    Returns:
        Tuple of (wavenumbers, crosssections) as float32 NumPy arrays.

    Raises:
        ImportError: If the `hapi` library is not installed.
    """
    try:
        import hapi  # type: ignore # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError("pip install hitran-api") from exc

    logger.info(
        "Querying HAPI cross-sections for %s across [%.1f, %.1f] cm^-1 at T=%.1fK, P=%.2fatm",
        gas_name,
        wavenumber_min,
        wavenumber_max,
        temperature,
        pressure,
    )
    # HAPI execution path
    table_name = f"hitran_{gas_name}"
    hapi.fetch(table_name, gas_name, 1, 1, wavenumber_min, wavenumber_max)
    nu, coef = hapi.absorptionCoefficient_Voigt(
        SourceTables=table_name,
        WavenumberRange=[wavenumber_min, wavenumber_max],
        Environment={"p": pressure, "T": temperature},
    )
    return np.array(nu, dtype=np.float32), np.array(coef, dtype=np.float32)


def apply_ils(
    spectrum: np.ndarray,
    resolution_cm: float = 40.0,
    delta_wavenumber: float = 10.0,
) -> np.ndarray:
    """Convolve spectrum with a sinc Instrument Line Shape (ILS) function.

    Args:
        spectrum: 1D absorption spectrum.
        resolution_cm: FTIR spectrometer spectral resolution in cm^-1 (default 40.0 cm^-1).
        delta_wavenumber: Step size between consecutive spectral channels in cm^-1.

    Returns:
        ILS-degraded 1D spectrum.
    """
    if len(spectrum) == 0:
        return spectrum

    n_pts = min(len(spectrum), 101)
    if n_pts % 2 == 0:
        n_pts += 1
    half = (n_pts - 1) // 2
    grid = np.arange(-half, half + 1, dtype=np.float32) * delta_wavenumber

    # Standard FTIR boxcar apodization produces sinc ILS: sinc(x) = sin(pi * x) / (pi * x)
    ils = np.sinc(grid / resolution_cm).astype(np.float32)
    ils_sum = np.sum(ils)
    if ils_sum > 0:
        ils = ils / ils_sum

    convolved = scipy.signal.fftconvolve(spectrum, ils, mode="same")
    return np.clip(convolved.astype(np.float32), 0.0, None)


def beer_lambert_mix(
    crosssections: Dict[str, np.ndarray],
    concentrations: Dict[str, float],
    path_length: float = 1.0,
) -> np.ndarray:
    """Compute composite optical absorbance using the linear Beer-Lambert law.

    A(nu) = sum_i c_i * sigma_i(nu) * L

    Args:
        crosssections: Mapping of gas names to absorption cross-section arrays.
        concentrations: Mapping of gas names to fractional or molar concentrations.
        path_length: Optical absorption path length (default 1.0).

    Returns:
        1D composite absorbance spectrum.
    """
    if not crosssections:
        return np.array([], dtype=np.float32)

    sample_arr = next(iter(crosssections.values()))
    absorbance = np.zeros_like(sample_arr, dtype=np.float32)

    for gas_name, conc in concentrations.items():
        if gas_name in crosssections:
            sigma = crosssections[gas_name]
            absorbance += (conc * sigma * path_length).astype(np.float32)

    return absorbance


def add_perturbations(
    spectrum: np.ndarray,
    snr: float,
    drift_amplitude: float = 0.01,
    drift_freq: float = 2.0,
    jitter_std: float = 0.02,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Add realistic instrumentation perturbations: baseline drift, jitter, and noise.

    Args:
        spectrum: Pure or convolved absorption spectrum.
        snr: Signal-to-Noise Ratio (ratio of peak signal to noise standard deviation).
        drift_amplitude: Maximum amplitude of smooth baseline drift.
        drift_freq: Frequency of sinusoidal baseline drift across spectral window.
        jitter_std: Standard deviation of channel wavenumber jitter.
        seed: Optional RNG seed.

    Returns:
        Perturbed spectrum array.
    """
    rng = np.random.RandomState(seed)
    n = len(spectrum)
    if n == 0:
        return spectrum

    # 1. Sinusoidal and low-order polynomial baseline drift
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    drift = drift_amplitude * np.sin(2.0 * np.pi * drift_freq * x + phase).astype(np.float32)
    poly_drift = rng.uniform(-drift_amplitude * 0.5, drift_amplitude * 0.5) * (x - 0.5)

    # 2. Additive Gaussian sensor noise based on SNR
    peak_val = float(np.max(spectrum)) if np.max(spectrum) > 1e-4 else 1.0
    noise_std = peak_val / max(snr, 1.0)
    noise = rng.normal(0.0, noise_std, size=n).astype(np.float32)

    # 3. Small channel jitter (interpolated shift)
    perturbed = spectrum + drift + poly_drift + noise
    if jitter_std > 0:
        jitter_shift = rng.normal(0.0, jitter_std)
        perturbed = np.interp(
            np.arange(n) + jitter_shift,
            np.arange(n),
            perturbed,
            left=perturbed[0],
            right=perturbed[-1],
        ).astype(np.float32)

    return np.clip(perturbed, 0.0, None)


def generate_synthetic_spectrum(
    gas_crosssections: Dict[str, np.ndarray],
    n_gases_range: Tuple[int, int] = (0, 10),
    concentration_range: Tuple[float, float] = (0.0, 1.0),
    resolution_cm: float = 40.0,
    snr_range: Tuple[float, float] = (500.0, 3000.0),
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Synthesize a single degraded multi-gas FTIR spectrum with full ground-truth metadata.

    Args:
        gas_crosssections: Dict mapping gas names to cross-section arrays.
        n_gases_range: Min and max count of simultaneously active gases (inclusive).
        concentration_range: (min, max) active gas concentration values.
        resolution_cm: FTIR spectrometer resolution in cm^-1.
        snr_range: (min, max) random SNR bounds.
        seed: Optional RNG seed.

    Returns:
        Tuple of (degraded_spectrum, metadata_dict).
    """
    rng = np.random.RandomState(seed)
    all_gases = list(gas_crosssections.keys())
    max_k = min(n_gases_range[1], len(all_gases))
    min_k = min(n_gases_range[0], max_k)
    k = rng.randint(min_k, max_k + 1) if max_k >= min_k else min_k

    selected_gases = list(rng.choice(all_gases, size=k, replace=False)) if k > 0 else []

    concentrations: Dict[str, float] = {}
    for gas in all_gases:
        if gas in selected_gases:
            concentrations[gas] = float(rng.uniform(concentration_range[0], concentration_range[1]))
        else:
            concentrations[gas] = 0.0

    # Optical linear superposition
    pure_mix = beer_lambert_mix(gas_crosssections, concentrations)

    # Instrument line shape convolution
    ils_spectrum = apply_ils(pure_mix, resolution_cm=resolution_cm)

    # Physical sensor perturbations
    snr = float(rng.uniform(snr_range[0], snr_range[1]))
    drift_amp = float(rng.uniform(0.005, 0.02))
    perturbed = add_perturbations(
        ils_spectrum,
        snr=snr,
        drift_amplitude=drift_amp,
        drift_freq=float(rng.uniform(1.0, 3.0)),
        seed=rng.randint(0, 1_000_000),
    )

    metadata: Dict[str, Any] = {
        "gases": selected_gases,
        "concentrations": concentrations,
        "snr": snr,
        "drift": drift_amp,
    }
    return perturbed, metadata


def generate_dataset(
    output_dir: str,
    num_samples: int,
    gas_names: List[str],
    wavenumber_range: Tuple[float, float] = (2000.0, 4000.0),
    resolution_cm: float = 40.0,
    seed: int = 42,
) -> Tuple[str, str, str, str]:
    """Generate a full synthetic FTIR dataset saved as memory-mapped binary and labeled subsets.

    Args:
        output_dir: Directory where generated files will be stored.
        num_samples: Total number of synthetic spectra to generate.
        gas_names: List of target gas identifiers.
        wavenumber_range: (min_cm-1, max_cm-1) spectral bounds.
        resolution_cm: Spectrometer resolution (cm^-1).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of paths: (mmap_path, labeled_npz_path, wavenumbers_path, metadata_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    # Compute wavenumber channels (e.g., 200 channels spanning 2000 to 4000 cm^-1, delta=10 cm^-1)
    step = 10.0
    num_channels = int(round((wavenumber_range[1] - wavenumber_range[0]) / step))
    wavenumbers = np.linspace(wavenumber_range[0], wavenumber_range[1], num_channels, dtype=np.float32)

    # Load or generate gas cross-sections
    crosssections: Dict[str, np.ndarray] = {}
    for gas in gas_names:
        try:
            _, sigma = fetch_hitran_crosssections(
                gas,
                wavenumber_range[0],
                wavenumber_range[1],
            )
            crosssections[gas] = sigma
        except (ImportError, Exception) as exc:
            logger.info("Using physical simulated cross-sections for %s (%s)", gas, exc)
            crosssections[gas] = _generate_offline_crosssection(gas, wavenumbers)

    logger.info(
        "Generating %d synthetic spectra (%d channels, %d gases)...",
        num_samples,
        num_channels,
        len(gas_names),
    )

    # 1. Create memmap for pre-training spectra
    mmap_path = os.path.join(output_dir, "pretrain_spectra.mmap")
    mmap_arr = np.memmap(mmap_path, dtype="float32", mode="w+", shape=(num_samples, num_channels))

    # Track labels and concentrations
    all_labels = np.zeros((num_samples, len(gas_names)), dtype=np.int32)
    all_concs = np.zeros((num_samples, len(gas_names)), dtype=np.float32)

    batch_size = 5000
    for start_idx in tqdm(range(0, num_samples, batch_size), desc="Synthesizing FTIR Spectra"):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_n = end_idx - start_idx
        batch_spectra = np.zeros((batch_n, num_channels), dtype=np.float32)

        for i in range(batch_n):
            spec_seed = rng.randint(0, 2_000_000_000)
            spec, meta = generate_synthetic_spectrum(
                crosssections,
                n_gases_range=(0, min(5, len(gas_names))),
                concentration_range=(0.05, 1.0),
                resolution_cm=resolution_cm,
                seed=spec_seed,
            )
            batch_spectra[i] = spec
            for g_idx, g_name in enumerate(gas_names):
                c = meta["concentrations"].get(g_name, 0.0)
                if c > 0.0:
                    all_labels[start_idx + i, g_idx] = 1
                    all_concs[start_idx + i, g_idx] = c

        mmap_arr[start_idx:end_idx] = batch_spectra

    mmap_arr.flush()
    del mmap_arr

    # 2. Save wavenumber axis
    wavenumbers_path = os.path.join(output_dir, "wavenumbers.npy")
    np.save(wavenumbers_path, wavenumbers)

    # 3. Save gas metadata JSON
    metadata_list = [
        {"name": g, "index": idx, "min_conc": 0.0, "max_conc": 1.0, "unit": "fraction"}
        for idx, g in enumerate(gas_names)
    ]
    metadata_path = os.path.join(output_dir, "gas_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2)

    # 4. Save labeled subset for downstream fine-tuning and evaluation
    labeled_path = os.path.join(output_dir, "labeled_spectra.npz")
    # Store at most 10,000 labeled samples or full set if smaller
    n_labeled = min(num_samples, 10000)
    labeled_mmap = np.memmap(mmap_path, dtype="float32", mode="r", shape=(num_samples, num_channels))
    np.savez(
        labeled_path,
        spectra=labeled_mmap[:n_labeled].copy(),
        gas_labels=all_labels[:n_labeled],
        concentrations=all_concs[:n_labeled],
        gas_names=np.array(gas_names),
    )
    del labeled_mmap

    logger.info("Synthetic dataset generated successfully at: %s", output_dir)
    return mmap_path, labeled_path, wavenumbers_path, metadata_path

