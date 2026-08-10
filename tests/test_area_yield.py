import numpy as np
import pytest
from src.actuarial_risk_model.area_yield import AreaYieldInsurance, load_yield_series


def test_load_yield_series_real_data() -> None:
    series = load_yield_series()
    assert len(series) > 50
    assert all(v > 0 for v in series.values())
    assert 1961 in series and 2023 in series


def test_fit_trend_on_synthetic_linear_series() -> None:
    yield_by_year = {y: 1000.0 + 20.0 * (y - 2000) for y in range(2000, 2020)}
    trend = AreaYieldInsurance.fit_trend(yield_by_year)
    assert trend['slope'] == pytest.approx(20.0, abs=1e-6)
    assert trend['intercept'] == pytest.approx(1000.0 - 20.0 * 2000, rel=1e-6)
    assert trend['residual_std'] == pytest.approx(0.0, abs=1e-6)


def test_trend_yield() -> None:
    trend = {'slope': 10.0, 'intercept': 500.0}
    assert AreaYieldInsurance.trend_yield(trend, 2020) == pytest.approx(500.0 + 10.0 * 2020)


def test_indemnity_zero_above_guarantee() -> None:
    assert AreaYieldInsurance.indemnity(actual_yield=1500, guaranteed_yield=1200, price_per_kg=50) == 0.0


def test_indemnity_positive_below_guarantee() -> None:
    result = AreaYieldInsurance.indemnity(actual_yield=1000, guaranteed_yield=1200, price_per_kg=50)
    assert result == pytest.approx(200 * 50)


def test_historical_indemnities_rejects_bad_coverage_level() -> None:
    with pytest.raises(ValueError):
        AreaYieldInsurance.historical_indemnities({2020: 1000.0}, {'slope': 0, 'intercept': 1200},
                                                    coverage_level=1.5, price_per_kg=50)


def test_historical_indemnities_shape() -> None:
    yield_by_year = {2020: 900.0, 2021: 1300.0}
    trend = {'slope': 0.0, 'intercept': 1200.0}
    result = AreaYieldInsurance.historical_indemnities(yield_by_year, trend, coverage_level=0.8, price_per_kg=50)
    # guaranteed = 0.8 * 1200 = 960
    assert result[2020] == pytest.approx((960 - 900) * 50)
    assert result[2021] == 0.0


def test_premium_from_indemnities() -> None:
    indemnities = np.array([0.0, 0.0, 5000.0, 0.0, 2000.0])
    result = AreaYieldInsurance.premium_from_indemnities(indemnities, risk_load=0.2, expense_load=0.15)
    assert result['pure_premium'] == pytest.approx(1400.0)
    assert result['gross_premium'] > result['pure_premium']
    assert 0 < result['loss_ratio'] < 1
