import pytest
from src.actuarial_risk_model.motor_insurance import MotorInsurancePricing, VEHICLE_CLASSES, BONUS_MALUS_SCALE


def test_base_premium_unknown_class() -> None:
    with pytest.raises(ValueError):
        MotorInsurancePricing.base_premium("spaceship")


def test_base_premium_matches_frequency_times_severity() -> None:
    result = MotorInsurancePricing.base_premium("private", risk_load=0.25, expense_load=0.2)
    vc = VEHICLE_CLASSES["private"]
    expected_pure = vc.annual_frequency * vc.mean_severity
    assert result['pure_premium'] == pytest.approx(expected_pure)
    assert result['gross_premium'] > result['pure_premium']


def test_psv_premium_exceeds_private() -> None:
    private = MotorInsurancePricing.base_premium("private")
    psv = MotorInsurancePricing.base_premium("psv")
    assert psv['gross_premium'] > private['gross_premium']


def test_bonus_malus_zero_years_no_discount() -> None:
    result = MotorInsurancePricing.bonus_malus_premium(10_000.0, claim_free_years=0)
    assert result['discount_pct'] == 0.0
    assert result['adjusted_premium'] == pytest.approx(10_000.0)


def test_bonus_malus_caps_at_max_scale() -> None:
    result = MotorInsurancePricing.bonus_malus_premium(10_000.0, claim_free_years=99)
    assert result['discount_pct'] == BONUS_MALUS_SCALE[max(BONUS_MALUS_SCALE)]
    assert result['adjusted_premium'] == pytest.approx(10_000.0 * (1 - result['discount_pct']))


def test_bonus_malus_negative_years_floors_at_zero_discount() -> None:
    result = MotorInsurancePricing.bonus_malus_premium(10_000.0, claim_free_years=-3)
    assert result['discount_pct'] == 0.0


def test_simulate_fleet_losses_shape_and_reproducibility() -> None:
    result_a = MotorInsurancePricing.simulate_fleet_losses("motorcycle", n_vehicles=500, n_years=1000, seed=7)
    result_b = MotorInsurancePricing.simulate_fleet_losses("motorcycle", n_vehicles=500, n_years=1000, seed=7)
    assert len(result_a['aggregate_losses']) == 1000
    assert (result_a['aggregate_losses'] == result_b['aggregate_losses']).all()
    assert result_a['aggregate_losses'].mean() > 0


def test_simulate_fleet_losses_unknown_class() -> None:
    with pytest.raises(ValueError):
        MotorInsurancePricing.simulate_fleet_losses("spaceship", n_vehicles=100, n_years=10)
