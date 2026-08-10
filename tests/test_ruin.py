import pytest
from src.actuarial_risk_model.ruin import RuinTheory


def test_adjustment_coefficient_exponential_closed_form() -> None:
    # for exponential claims, R = theta / (mu * (1+theta))
    claim_rate, mean, loading = 2.0, 1000.0, 0.2
    r = RuinTheory.adjustment_coefficient(claim_rate, 'exponential', {'mean': mean}, loading)
    expected = loading / (mean * (1 + loading))
    assert r == pytest.approx(expected, rel=1e-3)


def test_adjustment_coefficient_unsupported_severity() -> None:
    with pytest.raises(ValueError):
        RuinTheory.adjustment_coefficient(1.0, 'lognormal', {'mean': 100, 'std_dev': 50}, 0.2)


def test_ruin_probability_bound_decreases_with_surplus() -> None:
    r = 0.001
    p_low = RuinTheory.ruin_probability_bound(1000, r)
    p_high = RuinTheory.ruin_probability_bound(10_000, r)
    assert 0 < p_high < p_low <= 1


def test_ruin_probability_exact_matches_bound_direction() -> None:
    exact = RuinTheory.ruin_probability_exact_exponential(
        initial_surplus=5000, claim_rate=2.0, mean_severity=1000, premium_loading=0.2
    )
    r = RuinTheory.adjustment_coefficient(2.0, 'exponential', {'mean': 1000}, 0.2)
    bound = RuinTheory.ruin_probability_bound(5000, r)
    assert 0 < exact <= bound  # Lundberg's inequality must hold


def test_ruin_probability_requires_positive_loading() -> None:
    with pytest.raises(ValueError):
        RuinTheory.ruin_probability_exact_exponential(1000, 2.0, 1000, premium_loading=0)


def test_simulate_finite_horizon_ruin_reasonable() -> None:
    model = RuinTheory(seed=3)
    result = model.simulate_finite_horizon_ruin(
        initial_surplus=1000, claim_rate=2.0, severity_dist='exponential',
        severity_params={'mean': 1000}, premium_loading=0.2, time_horizon=5, n_paths=2_000
    )
    assert 0 <= result['ruin_probability'] <= 1
    assert result['n_paths'] == 2_000


def test_simulate_finite_horizon_ruin_zero_loading_higher_than_high_loading() -> None:
    model = RuinTheory(seed=3)
    low_loading = model.simulate_finite_horizon_ruin(
        1000, 2.0, 'exponential', {'mean': 1000}, premium_loading=0.05, time_horizon=10, n_paths=2_000
    )
    model2 = RuinTheory(seed=3)
    high_loading = model2.simulate_finite_horizon_ruin(
        1000, 2.0, 'exponential', {'mean': 1000}, premium_loading=1.0, time_horizon=10, n_paths=2_000
    )
    assert low_loading['ruin_probability'] >= high_loading['ruin_probability']
