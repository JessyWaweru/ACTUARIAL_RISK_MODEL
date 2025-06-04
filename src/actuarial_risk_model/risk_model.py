from typing import Self, TypedDict
import numpy as np
from scipy.stats import norm, poisson
import matplotlib.pyplot as plt

class DistributionParams(TypedDict):
    mean: float
    std_dev: float

class RiskModel:
    def __init__(self) -> None:
        self.simulations: int = 10_000  # default simulation count

    def monte_carlo_simulation(
        self, 
        dist_name: str, 
        params: DistributionParams
    ) -> np.ndarray[float]:  # ndarray stands for n dimensional array and is the fundamental data structure in NumPy for numerical computing
        # -> is like a return type annotation same as using {}
        """Enhanced with match-case (Python 3.10+)"""
        rng = np.random.default_rng()#modern random number generator
        
        match dist_name:
            case 'normal':
                return rng.normal(params['mean'], params['std_dev'], self.simulations)
            case 'poisson':
                return rng.poisson(params['mean'], self.simulations)
            case _:
                raise ValueError(f"Unsupported distribution: {dist_name}")

    def calculate_var(
        self, 
        losses: np.ndarray[float], 
        confidence: float = 0.95
    ) -> float:
        """Python 3.12 type parameter syntax"""
        if not 0 < confidence < 1:
            raise ValueError("Confidence must be between 0 and 1")
        return np.percentile(losses, 100 * confidence)