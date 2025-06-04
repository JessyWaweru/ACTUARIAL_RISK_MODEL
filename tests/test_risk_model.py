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