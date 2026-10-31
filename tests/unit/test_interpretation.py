"""Unit tests for gas scenarios, rule-based advisories, and GasAdvisor engine."""

from sfm.interpretation.llm import GasAdvisor
from sfm.interpretation.scenarios import (
    GAS_SCENARIOS,
    format_prediction_context,
    get_rule_based_advisory,
)


def test_gas_scenarios_coverage():
    """Verify standard FTIR target gases are covered in scenarios dictionary."""
    expected_gases = ["CO2", "CO", "CH4", "N2O", "H2O", "NH3", "SO2", "NO2"]
    for gas in expected_gases:
        assert gas in GAS_SCENARIOS
        assert "threshold_ppm" in GAS_SCENARIOS[gas]
        assert "warning" in GAS_SCENARIOS[gas]
        assert "danger" in GAS_SCENARIOS[gas]


def test_rule_based_advisory_levels():
    """Verify correct hazard status mapping for normal, warning, and danger concentrations."""
    # CO2: threshold=1000, danger=5000
    normal_adv = get_rule_based_advisory("CO2", 500.0)
    assert "within normal baseline" in normal_adv

    warning_adv = get_rule_based_advisory("CO2", 1500.0)
    assert "Consider improving ventilation" in warning_adv

    danger_adv = get_rule_based_advisory("CO2", 6000.0)
    assert "Evacuate area immediately" in danger_adv

    # Unknown gas
    unknown_adv = get_rule_based_advisory("Xenon", 10.0)
    assert "Unknown gas detected" in unknown_adv


def test_format_prediction_context():
    """Verify format_prediction_context string structure."""
    gas_names = ["CO2", "CO"]
    presence = [True, False]
    concs = [1200.0, 0.0]

    context = format_prediction_context(gas_names, presence, concs)
    assert "FTIR Multi-Gas Sensor Readings:" in context
    assert "CO2: DETECTED" in context
    assert "CO: NOT DETECTED" in context


def test_gas_advisor_fallback(monkeypatch):
    """Verify GasAdvisor generates valid text using rule-based engine when model offline."""
    # Force immediate fallback without network delay
    monkeypatch.setattr(GasAdvisor, "_try_load_model", lambda self: None)
    advisor = GasAdvisor(model_name="non_existent/model", device="cpu")
    assert not advisor.use_llm

    gas_names = ["CO2", "CH4"]
    presence = [True, False]
    concentrations = [5500.0, 10.0]

    advisory = advisor.generate_advisory(gas_names, presence, concentrations)
    assert isinstance(advisory, str)
    assert len(advisory) > 10
    assert "Safety Advisory" in advisory
    assert "Evacuate" in advisory

    # Test batch_interpret
    predictions = [
        {"gas_names": gas_names, "presence": presence, "concentrations": concentrations}
    ]
    annotated = advisor.batch_interpret(predictions)
    assert len(annotated) == 1
    assert "advisory" in annotated[0]
    assert annotated[0]["advisory"] == advisory
