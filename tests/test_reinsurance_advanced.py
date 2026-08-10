import pytest
from src.actuarial_risk_model.reinsurance_advanced import rate_on_line, price_xol_with_reinstatements


def test_rate_on_line() -> None:
    assert rate_on_line(premium=200_000, limit=1_000_000) == pytest.approx(0.2)


def test_rate_on_line_requires_positive_limit() -> None:
    with pytest.raises(ValueError):
        rate_on_line(premium=100, limit=0)


def test_price_xol_no_losses_gives_zero_recovery() -> None:
    years = [[100, 200], [50], []]
    result = price_xol_with_reinstatements(years, attachment=1_000_000, limit=5_000_000, num_reinstatements=2)
    assert result['expected_total_recovery'] == 0.0
    assert result['first_layer_pure_premium'] == 0.0
    assert result['expected_reinstatements_used'] == 0.0


def test_price_xol_single_large_loss_uses_layer() -> None:
    # one big occurrence loss eating into the layer, no reinstatement needed
    years = [[3_000_000], [0], [0]]
    result = price_xol_with_reinstatements(years, attachment=1_000_000, limit=5_000_000, num_reinstatements=1)
    assert result['expected_total_recovery'] == pytest.approx((2_000_000) / 3)
    assert result['expected_reinstatements_used'] == pytest.approx(0.0)
    assert result['gross_premium'] == pytest.approx(result['first_layer_premium'])  # no reinstatement consumed


def test_price_xol_reinstatement_triggered() -> None:
    # two occurrences in one year both exhausting the first layer -> 1 reinstatement used
    years = [[6_000_000, 6_000_000]]
    result = price_xol_with_reinstatements(years, attachment=1_000_000, limit=5_000_000, num_reinstatements=1)
    # each occurrence layer loss = 5,000,000 (capped at limit), total = 10,000,000 = full aggregate capacity
    assert result['expected_total_recovery'] == pytest.approx(10_000_000)
    assert result['expected_reinstatements_used'] == pytest.approx(1.0)


def test_price_xol_negative_reinstatements_raises() -> None:
    with pytest.raises(ValueError):
        price_xol_with_reinstatements([[100]], attachment=0, limit=100, num_reinstatements=-1)
