"""
Health microinsurance: claims reserving (IBNR) via Mack's stochastic chain
ladder, plus catastrophic/stop-loss cover pricing for a scheme's aggregate
annual claims.

The illustrative claims triangle and per-scheme claim volume are calibrated
to typical microinsurance scheme scale (a growing membership base, a health
claims development pattern that's mostly reported within its accident year)
rather than a real scheme's book, since granular microinsurance claims data
isn't publicly available.
"""
from typing import Dict, List, Optional
import numpy as np

from .reserving import mack_chain_ladder
from .risk_model import RiskModel


def build_illustrative_triangle(n_years: int = 5, base_claims: float = 800_000,
                                 growth: float = 0.12, dev_pattern: Optional[List[float]] = None,
                                 seed: Optional[int] = None) -> np.ndarray:
    """
    An n_years x n_years cumulative-claims triangle for a growing
    microinsurance scheme: membership growth drives up each accident year's
    ultimate claims, developed on a typical health-claims pattern (~55%
    reported by the end of the accident year, fully developed by year 5).
    """
    if dev_pattern is None:
        dev_pattern = [0.55, 0.80, 0.92, 0.98, 1.0]
    dev_pattern = list(dev_pattern[:n_years])
    dev_pattern[-1] = 1.0

    rng = np.random.default_rng(seed)
    ultimate = base_claims * (1 + growth) ** np.arange(n_years) * rng.normal(1.0, 0.05, n_years)

    triangle = np.full((n_years, n_years), np.nan)
    for i in range(n_years):
        for j in range(n_years - i):
            triangle[i, j] = ultimate[i] * dev_pattern[j]
    return triangle


class HealthMicroReserving:

    @staticmethod
    def reserve(triangle: np.ndarray) -> Dict[str, np.ndarray]:
        """IBNR via Mack's stochastic chain ladder (reserving.mack_chain_ladder)."""
        return mack_chain_ladder(triangle)

    @staticmethod
    def simulate_annual_claims(mean_annual_claims: float, cv: float, n_years: int,
                                seed: Optional[int] = None) -> np.ndarray:
        """Simulate the scheme's aggregate annual claims via a lognormal (cv = coefficient of variation)."""
        rng = np.random.default_rng(seed)
        std = mean_annual_claims * cv
        sigma2 = np.log(1 + cv ** 2)
        mu = np.log(mean_annual_claims) - sigma2 / 2
        return rng.lognormal(mu, np.sqrt(sigma2), n_years)

    @staticmethod
    def catastrophic_cover_pricing(annual_claims: np.ndarray, deductible: float, limit: float) -> Dict[str, float]:
        """
        Price a stop-loss layer over the scheme's aggregate annual claims:
        the scheme retains claims up to `deductible`, a catastrophic cover
        layer picks up the next `limit` (reuses risk_model's occurrence
        layer formula at the aggregate-annual level).
        """
        model = RiskModel()
        return model.price_reinsurance_layer(np.asarray(annual_claims, dtype=float), deductible, limit)
