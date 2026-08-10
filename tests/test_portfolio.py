import numpy as np
import pytest
from src.actuarial_risk_model.portfolio import CorrelatedPortfolio

LINES = [
    {'name': 'property', 'dist': 'lognormal', 'mean': 1_000_000, 'std_dev': 300_000},
    {'name': 'liability', 'dist': 'gamma', 'mean': 500_000, 'std_dev': 200_000},
]


def test_simulate_shapes() -> None:
    model = CorrelatedPortfolio(seed=1)
    result = model.simulate(LINES, correlation_matrix=np.array([[1.0, 0.3], [0.3, 1.0]]), n_simulations=5_000)
    assert result['total_losses'].shape == (5_000,)
    assert result['per_line_losses'].shape == (5_000, 2)
    assert result['line_names'] == ['property', 'liability']


def test_perfect_correlation_gives_zero_diversification_benefit() -> None:
    model = CorrelatedPortfolio(seed=1)
    result = model.simulate(LINES, correlation_matrix=np.array([[1.0, 1.0], [1.0, 1.0]]), n_simulations=20_000)
    assert result['diversification_benefit_pct'] == pytest.approx(0.0, abs=2.0)


def test_low_correlation_gives_positive_diversification_benefit() -> None:
    model = CorrelatedPortfolio(seed=1)
    result = model.simulate(LINES, correlation_matrix=np.array([[1.0, 0.0], [0.0, 1.0]]), n_simulations=20_000)
    assert result['diversification_benefit_pct'] > 5.0


def test_mismatched_correlation_matrix_raises() -> None:
    model = CorrelatedPortfolio(seed=1)
    with pytest.raises(ValueError):
        model.simulate(LINES, correlation_matrix=np.eye(3), n_simulations=100)


def test_non_psd_correlation_matrix_raises() -> None:
    model = CorrelatedPortfolio(seed=1)
    bad_corr = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError):
        model.simulate(LINES, correlation_matrix=bad_corr, n_simulations=100)
