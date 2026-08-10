import pytest
from src.actuarial_risk_model.risk_model import RiskModel
import numpy as np

@pytest.fixture
def model() -> RiskModel:
    return RiskModel()

def test_var_calculation(model: RiskModel) -> None:
    test_data = np.array([1, 2, 3, 4, 5])
    assert model.calculate_var(test_data, 0.8) == pytest.approx(4.2)  # Allows floating-point tolerance

def test_simulation_shape(model: RiskModel) -> None:
    losses = model.monte_carlo_simulation('normal', {'mean': 0, 'std_dev': 1})
    assert losses.shape == (10_000,)

def test_chain_ladder_reserve(model: RiskModel) -> None:
    triangle = np.array([
        [100.0, 150.0, 175.0, 180.0],
        [120.0, 180.0, 200.0, np.nan],
        [140.0, 200.0, np.nan, np.nan],
        [130.0, np.nan, np.nan, np.nan],
    ])
    total_reserve, dev_factors = model.chain_ladder_reserve(triangle)
    assert total_reserve > 0
    assert dev_factors[0] == pytest.approx(530 / 360)
    assert dev_factors[1] == pytest.approx(375 / 330)
    assert dev_factors[2] == pytest.approx(180 / 175)