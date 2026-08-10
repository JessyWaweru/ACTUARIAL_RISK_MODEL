import pytest
from src.actuarial_risk_model.credibility import BuhlmannCredibility


def test_estimate_parameters_basic() -> None:
    claims = [
        [100, 120, 110],
        [500, 520, 480],
        [200, 210, 190],
    ]
    params = BuhlmannCredibility.estimate_parameters(claims)
    assert params['epv'] > 0
    assert params['vhm'] >= 0
    assert params['collective_mean'] == pytest.approx((110 + 500 + 200) / 3, rel=0.2)


def test_estimate_parameters_requires_multiple_risks() -> None:
    with pytest.raises(ValueError):
        BuhlmannCredibility.estimate_parameters([[100, 200]])


def test_credibility_factor_bounds() -> None:
    z_small_n = BuhlmannCredibility.credibility_factor(n=1, epv=100, vhm=1)
    z_large_n = BuhlmannCredibility.credibility_factor(n=100_000, epv=100, vhm=1)
    assert 0 <= z_small_n < z_large_n <= 1
    assert z_large_n == pytest.approx(1.0, abs=0.01)


def test_credibility_factor_zero_vhm() -> None:
    assert BuhlmannCredibility.credibility_factor(n=50, epv=10, vhm=0) == 0.0


def test_credibility_premium_blends_toward_collective_mean() -> None:
    result = BuhlmannCredibility.credibility_premium(
        individual_mean=1000, n=1, epv=500, vhm=1, collective_mean=100
    )
    assert 100 < result['credibility_premium'] < 1000
    assert 0 <= result['z'] <= 1
