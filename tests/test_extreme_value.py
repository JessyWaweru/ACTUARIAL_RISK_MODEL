import numpy as np
import pytest
from src.actuarial_risk_model.extreme_value import ExtremeValueModel


@pytest.fixture
def exponential_losses() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.exponential(scale=1000, size=5_000)


def test_fit_gpd_requires_enough_exceedances() -> None:
    losses = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        ExtremeValueModel.fit_gpd(losses, threshold=10)


def test_fit_gpd_basic(exponential_losses: np.ndarray) -> None:
    result = ExtremeValueModel.fit_gpd(exponential_losses, threshold=1000)
    assert result['n_exceedances'] > 10
    assert 0 < result['exceedance_rate'] < 1
    assert result['scale'] > 0
    # exponential tail => GPD shape parameter should fit near 0
    assert abs(result['shape']) < 0.3


def test_return_level_increases_with_period() -> None:
    level_20 = ExtremeValueModel.return_level(shape=0.1, scale=500, threshold=1000,
                                               exceedance_rate=0.1, return_period=20)
    level_200 = ExtremeValueModel.return_level(shape=0.1, scale=500, threshold=1000,
                                                exceedance_rate=0.1, return_period=200)
    assert level_200 > level_20 > 1000


def test_tail_risk_metrics_tvar_exceeds_var() -> None:
    result = ExtremeValueModel.tail_risk_metrics(shape=0.2, scale=500, threshold=1000,
                                                  exceedance_rate=0.1, confidence=0.99)
    assert result['tvar'] > result['var'] > 1000


def test_tail_risk_metrics_rejects_low_confidence() -> None:
    with pytest.raises(ValueError):
        ExtremeValueModel.tail_risk_metrics(shape=0.2, scale=500, threshold=1000,
                                             exceedance_rate=0.05, confidence=0.5)
