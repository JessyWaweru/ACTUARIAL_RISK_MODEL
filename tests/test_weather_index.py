import numpy as np
import pytest
from src.actuarial_risk_model.weather_index import WeatherIndexInsurance, load_monthly_rainfall, COUNTIES


def test_load_monthly_rainfall_real_data() -> None:
    for county in COUNTIES:
        rows = load_monthly_rainfall(county)
        assert len(rows) > 300  # 1991-2023 monthly
        assert all(r['rainfall_mm'] >= 0 for r in rows)
        assert {row['month'] for row in rows} == set(range(1, 13))


def test_load_monthly_rainfall_unknown_county() -> None:
    with pytest.raises(ValueError):
        load_monthly_rainfall("nairobi")


def test_seasonal_index_sums_requested_months() -> None:
    rows = [
        {'year': 2020, 'month': 3, 'rainfall_mm': 10.0},
        {'year': 2020, 'month': 4, 'rainfall_mm': 20.0},
        {'year': 2020, 'month': 5, 'rainfall_mm': 30.0},
        {'year': 2021, 'month': 3, 'rainfall_mm': 5.0},
        # 2021 missing April/May -> excluded
    ]
    index = WeatherIndexInsurance.seasonal_index(rows, months=[3, 4, 5])
    assert index == {2020: 60.0}


def test_payout_fraction_scalar_bounds() -> None:
    assert WeatherIndexInsurance.payout_fraction(200, strike=150, exit_level=50) == 0.0
    assert WeatherIndexInsurance.payout_fraction(50, strike=150, exit_level=50) == 1.0
    assert WeatherIndexInsurance.payout_fraction(0, strike=150, exit_level=50) == 1.0
    mid = WeatherIndexInsurance.payout_fraction(100, strike=150, exit_level=50)
    assert mid == pytest.approx(0.5)


def test_payout_fraction_vectorized() -> None:
    values = np.array([200, 100, 0])
    fracs = WeatherIndexInsurance.payout_fraction(values, strike=150, exit_level=50)
    assert isinstance(fracs, np.ndarray)
    np.testing.assert_allclose(fracs, [0.0, 0.5, 1.0])


def test_payout_fraction_rejects_bad_levels() -> None:
    with pytest.raises(ValueError):
        WeatherIndexInsurance.payout_fraction(100, strike=50, exit_level=150)


def test_historical_payouts() -> None:
    index_by_year = {2020: 200.0, 2021: 100.0, 2022: 0.0}
    payouts = WeatherIndexInsurance.historical_payouts(index_by_year, strike=150, exit_level=50, sum_insured=1000)
    assert payouts == {2020: 0.0, 2021: 500.0, 2022: 1000.0}


def test_burn_cost_premium_no_variance_no_payouts() -> None:
    result = WeatherIndexInsurance.burn_cost_premium(np.zeros(10))
    assert result['pure_premium'] == 0.0
    assert result['gross_premium'] == 0.0


def test_burn_cost_premium_positive_payouts() -> None:
    payouts = np.array([0.0, 0.0, 1000.0, 0.0, 500.0])
    result = WeatherIndexInsurance.burn_cost_premium(payouts, risk_load=0.2, expense_load=0.15)
    assert result['pure_premium'] == pytest.approx(300.0)
    assert result['gross_premium'] > result['pure_premium']
    assert 0 < result['loss_ratio'] < 1
