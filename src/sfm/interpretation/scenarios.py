"""Gas safety thresholds, hazard scenarios, and prompt formatting templates for FTIR gas sensing.

Defines:
- GAS_SCENARIOS: Domain dictionary of environmental and occupational exposure thresholds.
- get_rule_based_advisory: Deterministic fallback logic mapping gas concentrations to advisory actions.
- format_prediction_context: Formatter converting raw multi-gas model outputs into structured prompt contexts.
"""

from __future__ import annotations

from typing import Any, Dict, List

GAS_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "CO2": {
        "threshold_ppm": 1000.0,
        "danger_ppm": 5000.0,
        "warning": "CO2 concentration elevated. Consider improving ventilation.",
        "danger": "CO2 concentration dangerously high. Evacuate area immediately.",
        "unit": "ppm",
    },
    "CO": {
        "threshold_ppm": 35.0,
        "danger_ppm": 100.0,
        "warning": "Carbon monoxide detected. Check for combustion sources.",
        "danger": "Dangerous CO levels. Evacuate immediately and seek medical attention.",
        "unit": "ppm",
    },
    "CH4": {
        "threshold_ppm": 5000.0,
        "danger_ppm": 15000.0,
        "warning": "Methane detected. Check for gas leaks.",
        "danger": "Methane at explosive concentration. Evacuate.",
        "unit": "ppm",
    },
    "N2O": {
        "threshold_ppm": 25.0,
        "danger_ppm": 100.0,
        "warning": "Nitrous oxide detected above occupational limit.",
        "danger": "N2O at harmful levels.",
        "unit": "ppm",
    },
    "H2O": {
        "threshold_ppm": 80.0,
        "danger_ppm": 95.0,
        "warning": "Elevated humidity detected.",
        "danger": "Excessive humidity and condensation risk.",
        "unit": "%RH",
    },
    "NH3": {
        "threshold_ppm": 25.0,
        "danger_ppm": 50.0,
        "warning": "Ammonia detected above occupational threshold.",
        "danger": "Dangerous ammonia levels. Eye and respiratory hazard. Evacuate.",
        "unit": "ppm",
    },
    "SO2": {
        "threshold_ppm": 5.0,
        "danger_ppm": 20.0,
        "warning": "Sulfur dioxide detected. Respiratory irritant.",
        "danger": "Severe SO2 hazard. Immediate evacuation required.",
        "unit": "ppm",
    },
    "NO2": {
        "threshold_ppm": 3.0,
        "danger_ppm": 10.0,
        "warning": "Nitrogen dioxide elevated. Potential respiratory risk.",
        "danger": "High NO2 toxic gas concentration. Evacuate area.",
        "unit": "ppm",
    },
    "VOC": {
        "threshold_ppm": 0.5,
        "danger_ppm": 3.0,
        "warning": "Volatile organic compounds detected above normal baseline.",
        "danger": "Hazardous VOC concentration. Ventilate and investigate source.",
        "unit": "ppm",
    },
}


def get_rule_based_advisory(gas_name: str, concentration: float) -> str:
    """Return a deterministic hazard advisory for a detected gas and its concentration.

    Args:
        gas_name: Name or chemical formula of the gas (e.g. 'CO2', 'CH4').
        concentration: Estimated concentration in appropriate units (ppm or %RH).

    Returns:
        Actionable safety advisory string.
    """
    clean_name = gas_name.strip()
    scenario = None
    for k, v in GAS_SCENARIOS.items():
        if k.lower() == clean_name.lower():
            scenario = v
            break

    if scenario is None:
        return f"Unknown gas detected: {gas_name} at concentration {concentration:.2f}."

    thresh = scenario.get("threshold_ppm", 100.0)
    danger_thresh = scenario.get("danger_ppm", thresh * 2.0 if thresh else 200.0)
    unit = scenario.get("unit", "ppm")

    if danger_thresh is not None and concentration >= danger_thresh:
        return scenario["danger"]
    elif thresh is not None and concentration >= thresh:
        return scenario["warning"]
    else:
        return f"{clean_name} concentration within normal baseline ({concentration:.2f} {unit})."


def format_prediction_context(
    gas_names: List[str],
    presence: List[bool],
    concentrations: List[float],
) -> str:
    """Format multi-gas predictions into a structured text prompt for the LLM.

    Args:
        gas_names: List of target gas identifiers.
        presence: Boolean presence flag for each gas.
        concentrations: Numerical concentration values for each gas.

    Returns:
        Structured context block ready for inclusion in LLM prompt templates.
    """
    lines = ["FTIR Multi-Gas Sensor Readings:"]
    for name, is_present, conc in zip(gas_names, presence, concentrations):
        unit = GAS_SCENARIOS.get(name, {}).get("unit", "ppm")
        status = "DETECTED" if is_present else "NOT DETECTED"
        lines.append(f"- {name}: {status} | Concentration: {conc:.2f} {unit}")
    return "\n".join(lines)

