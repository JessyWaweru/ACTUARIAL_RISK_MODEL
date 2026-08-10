import numpy as np
import pytest
from src.actuarial_risk_model.cat_bond import FloodCatBond, flood_payout_fraction, load_daily_rainfall
from src.actuarial_risk_model.extreme_value import ExtremeValueModel


def test_load_daily_rainfall_real_data() -> None:
    rainfall = load_daily_rainfall()
    assert len(rainfall) > 8000  # 2001-2023 daily
    assert all(v >= 0 for v in rainfall)


def test_flood_payout_fraction_scalar_bounds() -> None:
    assert flood_payout_fraction(10, attachment_mm=50, exhaustion_mm=150) == 0.0
    assert flood_payout_fraction(50, attachment_mm=50, exhaustion_mm=150) == 0.0
    assert flood_payout_fraction(150, attachment_mm=50, exhaustion_mm=150) == 1.0
    assert flood_payout_fraction(200, attachment_mm=50, exhaustion_mm=150) == 1.0
    assert flood_payout_fraction(100, attachment_mm=50, exhaustion_mm=150) == pytest.approx(0.5)


def test_flood_payout_fraction_vectorized() -> None:
    values = np.array([10, 100, 200])
    fracs = flood_payout_fraction(values, attachment_mm=50, exhaustion_mm=150)
    assert isinstance(fracs, np.ndarray)
    np.testing.assert_allclose(fracs, [0.0, 0.5, 1.0])


def test_flood_payout_fraction_rejects_bad_levels() -> None:
    with pytest.raises(ValueError):
        flood_payout_fraction(100, attachment_mm=150, exhaustion_mm=50)


def test_simulate_annual_max_rainfall_on_real_data() -> None:
    rainfall = np.array(load_daily_rainfall())
    threshold = float(np.percentile(rainfall, 98))
    fit = ExtremeValueModel.fit_gpd(rainfall, threshold)
    simulated = FloodCatBond.simulate_annual_max_rainfall(
        fit['shape'], fit['scale'], fit['threshold'], fit['exceedance_rate'], n_years=2000, seed=42
    )
    assert len(simulated) == 2000
    assert np.all(simulated >= threshold)
    # Some years should see rainfall well above threshold (that's the point of the tail)
    assert simulated.max() > threshold


def test_price_bond_basic() -> None:
    # Half the simulated years exceed attachment, none reach exhaustion
    annual_max = np.array([10.0, 60.0, 80.0, 10.0, 70.0])
    result = FloodCatBond.price_bond(annual_max, attachment_mm=50, exhaustion_mm=200, principal=1_000_000)
    assert 0 < result['expected_loss_pct'] < 1
    assert result['probability_of_attachment'] == pytest.approx(0.6)
    assert result['probability_of_exhaustion'] == 0.0
    assert result['coupon_spread_pct'] == pytest.approx(result['expected_loss_pct'] * 3.0)
    assert result['expected_loss_amount'] == pytest.approx(result['expected_loss_pct'] * 1_000_000)


def test_price_bond_full_exhaustion() -> None:
    annual_max = np.array([500.0, 500.0])
    result = FloodCatBond.price_bond(annual_max, attachment_mm=50, exhaustion_mm=200, principal=1_000_000)
    assert result['expected_loss_pct'] == pytest.approx(1.0)
    assert result['probability_of_exhaustion'] == pytest.approx(1.0)
