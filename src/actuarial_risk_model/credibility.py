"""
Buhlmann credibility theory: blend individual risk experience with the
collective (portfolio) mean, weighted by a credibility factor Z derived from
the expected process variance (EPV) and variance of hypothetical means (VHM).
"""
from typing import Dict, List, Sequence
import numpy as np


class BuhlmannCredibility:

    @staticmethod
    def estimate_parameters(claims_by_risk: Sequence[Sequence[float]]) -> Dict[str, float]:
        """
        Estimate Buhlmann's EPV, VHM and collective mean from historical data.

        Args:
            claims_by_risk: one sequence of observed claim amounts (or loss
                ratios) per risk/policyholder, across periods.
        """
        groups = [np.asarray(c, dtype=float) for c in claims_by_risk]
        n_risks = len(groups)
        if n_risks < 2:
            raise ValueError("Need at least 2 risks to estimate credibility parameters")

        ns = np.array([len(c) for c in groups], dtype=float)
        if np.any(ns < 1):
            raise ValueError("Every risk needs at least 1 observation")

        means = np.array([c.mean() for c in groups])
        variances = np.array([c.var(ddof=1) if len(c) > 1 else 0.0 for c in groups])

        epv = float(np.average(variances, weights=ns))
        collective_mean = float(np.average(means, weights=ns))

        n_bar = float(ns.mean())
        if n_risks > 1:
            weighted_var_of_means = float(np.sum(ns * (means - collective_mean) ** 2) / (n_risks - 1))
        else:
            weighted_var_of_means = 0.0
        vhm = max((weighted_var_of_means - epv) / n_bar, 0.0)

        return {'epv': epv, 'vhm': vhm, 'collective_mean': collective_mean}

    @staticmethod
    def credibility_factor(n: float, epv: float, vhm: float) -> float:
        """Z = n / (n + k), k = EPV / VHM. Z=0 if VHM is 0 (no evidence of heterogeneity)."""
        if vhm <= 0:
            return 0.0
        k = epv / vhm
        return n / (n + k)

    @staticmethod
    def credibility_premium(individual_mean: float,
                             n: float,
                             epv: float,
                             vhm: float,
                             collective_mean: float) -> Dict[str, float]:
        """Credibility-weighted premium: Z * individual_mean + (1-Z) * collective_mean."""
        z = BuhlmannCredibility.credibility_factor(n, epv, vhm)
        premium = z * individual_mean + (1 - z) * collective_mean
        return {'z': z, 'credibility_premium': premium}
