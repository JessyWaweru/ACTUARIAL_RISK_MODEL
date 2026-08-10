import numpy as np
import pytest
from src.actuarial_risk_model.health_micro import (
    HealthMicroReserving, build_illustrative_triangle,
)


def test_build_illustrative_triangle_shape() -> None:
    triangle = build_illustrative_triangle(n_years=5, seed=1)
    assert triangle.shape == (5, 5)
    # upper-left is fully populated (accident year 0 through all dev periods)
    assert not np.isnan(triangle[0]).any()
    # last accident year only has its first development period
    assert not np.isnan(triangle[-1, 0])
    assert np.isnan(triangle[-1, 1:]).all()


def test_build_illustrative_triangle_reproducible_with_seed() -> None:
    a = build_illustrative_triangle(n_years=4, seed=5)
    b = build_illustrative_triangle(n_years=4, seed=5)
    np.testing.assert_allclose(a, b, equal_nan=True)


def test_reserve_returns_positive_total() -> None:
    triangle = build_illustrative_triangle(n_years=5, seed=3)
    result = HealthMicroReserving.reserve(triangle)
    assert result['total_reserve'] > 0
    assert result['total_standard_error'] >= 0
    assert len(result['dev_factors']) == 4


def test_simulate_annual_claims_positive_and_reproducible() -> None:
    a = HealthMicroReserving.simulate_annual_claims(mean_annual_claims=1_000_000, cv=0.3, n_years=500, seed=9)
    b = HealthMicroReserving.simulate_annual_claims(mean_annual_claims=1_000_000, cv=0.3, n_years=500, seed=9)
    assert len(a) == 500
    assert (a > 0).all()
    np.testing.assert_allclose(a, b)
    assert a.mean() == pytest.approx(1_000_000, rel=0.1)


def test_catastrophic_cover_pricing() -> None:
    claims = np.array([500_000, 1_500_000, 800_000, 2_500_000, 600_000])
    result = HealthMicroReserving.catastrophic_cover_pricing(claims, deductible=1_000_000, limit=2_000_000)
    assert result['pure_premium'] > 0
    assert result['gross_premium'] > result['pure_premium']
    assert 0 <= result['loss_ratio'] <= 1
