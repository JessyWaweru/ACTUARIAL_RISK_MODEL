import numpy as np
import pytest
from src.actuarial_risk_model.reserving import mack_chain_ladder, bornhuetter_ferguson

TRIANGLE = np.array([
    [100.0, 150.0, 175.0, 180.0],
    [120.0, 180.0, 200.0, np.nan],
    [140.0, 200.0, np.nan, np.nan],
    [130.0, np.nan, np.nan, np.nan],
])


def test_mack_dev_factors() -> None:
    result = mack_chain_ladder(TRIANGLE)
    assert result['dev_factors'][0] == pytest.approx(530 / 360)
    assert result['dev_factors'][1] == pytest.approx(375 / 330)
    assert result['dev_factors'][2] == pytest.approx(180 / 175)


def test_mack_fully_developed_year_has_zero_reserve() -> None:
    result = mack_chain_ladder(TRIANGLE)
    assert result['ultimate_by_year'][0] == pytest.approx(180.0)
    assert result['reserve_by_year'][0] == pytest.approx(0.0)
    assert result['standard_error_by_year'][0] == pytest.approx(0.0)


def test_mack_reserves_non_negative_and_consistent() -> None:
    result = mack_chain_ladder(TRIANGLE)
    assert np.all(result['reserve_by_year'] >= 0)
    assert result['total_reserve'] == pytest.approx(np.sum(result['reserve_by_year']))
    assert result['total_standard_error'] >= 0


def test_bornhuetter_ferguson_consistent_with_diagonal() -> None:
    premium = np.array([1000.0, 1100.0, 1200.0, 1300.0])
    result = bornhuetter_ferguson(TRIANGLE, expected_loss_ratio=0.6, premium=premium)

    latest_diagonal = np.array([TRIANGLE[i, 3 - i] for i in range(4)])
    assert result['ultimate_by_year'] == pytest.approx(latest_diagonal + result['reserve_by_year'])
    assert result['pct_reported_by_year'][0] == pytest.approx(1.0)  # fully developed year
    assert result['reserve_by_year'][0] == pytest.approx(0.0)
    assert result['total_reserve'] == pytest.approx(np.sum(result['reserve_by_year']))


def test_bornhuetter_ferguson_requires_matching_premium_length() -> None:
    with pytest.raises(ValueError):
        bornhuetter_ferguson(TRIANGLE, expected_loss_ratio=0.6, premium=np.array([1000.0, 1100.0]))
