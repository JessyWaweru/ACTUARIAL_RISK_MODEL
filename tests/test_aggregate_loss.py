import numpy as np
import pytest
from src.actuarial_risk_model.aggregate_loss import AggregateLossModel


@pytest.fixture
def model() -> AggregateLossModel:
    return AggregateLossModel(seed=42)


def test_simulate_shape(model: AggregateLossModel) -> None:
    result = model.simulate('poisson', {'mean': 3}, 'lognormal', {'mean': 1000, 'std_dev': 500}, n_simulations=5_000)
    assert result['aggregate_losses'].shape == (5_000,)
    assert result['claim_counts'].shape == (5_000,)


def test_zero_claims_gives_zero_loss(model: AggregateLossModel) -> None:
    result = model.simulate('poisson', {'mean': 0.0}, 'gamma', {'mean': 100, 'std_dev': 50}, n_simulations=1_000)
    assert np.all(result['claim_counts'] == 0)
    assert np.all(result['aggregate_losses'] == 0)


def test_mean_aggregate_loss_approx_frequency_times_severity(model: AggregateLossModel) -> None:
    freq_mean, sev_mean = 5.0, 200.0
    result = model.simulate('poisson', {'mean': freq_mean}, 'gamma', {'mean': sev_mean, 'std_dev': 50},
                             n_simulations=50_000)
    expected = freq_mean * sev_mean
    assert result['aggregate_losses'].mean() == pytest.approx(expected, rel=0.05)


def test_negative_binomial_requires_overdispersion(model: AggregateLossModel) -> None:
    with pytest.raises(ValueError):
        model.simulate('negative_binomial', {'mean': 3, 'dispersion': 0.5}, 'gamma', {'mean': 100, 'std_dev': 50})


def test_unsupported_distributions_raise(model: AggregateLossModel) -> None:
    with pytest.raises(ValueError):
        model.simulate('bogus', {'mean': 1}, 'gamma', {'mean': 1, 'std_dev': 1})
    with pytest.raises(ValueError):
        model.simulate('poisson', {'mean': 1}, 'bogus', {'mean': 1, 'std_dev': 1})
