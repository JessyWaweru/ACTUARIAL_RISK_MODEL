"""
Compound (collective risk model) aggregate loss simulation.

Unlike a single-distribution Monte Carlo run, this simulates claim *frequency*
and claim *severity* separately and sums severities per period, which is the
standard actuarial approach to modeling aggregate annual losses.
"""
from typing import Dict, Optional
import numpy as np


class AggregateLossModel:
    """Frequency x severity compound Monte Carlo simulation."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def _draw_frequency(self, dist_name: str, params: Dict[str, float], n: int) -> np.ndarray:
        if dist_name == 'poisson':
            return self.rng.poisson(params['mean'], n)
        elif dist_name == 'negative_binomial':
            mean = params['mean']
            dispersion = params.get('dispersion', 1.5)
            if dispersion <= 1:
                raise ValueError(
                    "dispersion must be > 1 for negative binomial "
                    "(it must be overdispersed relative to Poisson, where dispersion=1)"
                )
            variance = mean * dispersion
            p = mean / variance
            r = mean * p / (1 - p)
            return self.rng.negative_binomial(r, p, n)
        else:
            raise ValueError(f"Unsupported frequency distribution: {dist_name}")

    def _draw_severity(self, dist_name: str, params: Dict[str, float], n: int) -> np.ndarray:
        if n == 0:
            return np.array([])
        if dist_name == 'lognormal':
            mean, std = params['mean'], params['std_dev']
            sigma2 = np.log(1 + (std ** 2 / mean ** 2))
            mu = np.log(mean) - sigma2 / 2
            return self.rng.lognormal(mu, np.sqrt(sigma2), n)
        elif dist_name == 'gamma':
            mean, std = params['mean'], params['std_dev']
            shape = mean ** 2 / std ** 2
            scale = std ** 2 / mean
            return self.rng.gamma(shape, scale, n)
        elif dist_name == 'pareto':
            # Pareto Type I with support [scale, inf), shape=alpha (tail index)
            alpha, scale = params['alpha'], params['scale']
            return (self.rng.pareto(alpha, n) + 1) * scale
        else:
            raise ValueError(f"Unsupported severity distribution: {dist_name}")

    def simulate(self,
                 freq_dist: str,
                 freq_params: Dict[str, float],
                 sev_dist: str,
                 sev_params: Dict[str, float],
                 n_simulations: int = 10_000) -> Dict[str, np.ndarray]:
        """
        Simulate `n_simulations` independent periods (e.g. policy years).

        Returns a dict with:
        - aggregate_losses: total loss per simulated period
        - claim_counts: number of claims per simulated period
        """
        counts = self._draw_frequency(freq_dist, freq_params, n_simulations)
        total_claims = int(counts.sum())
        severities = self._draw_severity(sev_dist, sev_params, total_claims)

        aggregate = np.zeros(n_simulations)
        if total_claims > 0:
            period_idx = np.repeat(np.arange(n_simulations), counts)
            aggregate = np.bincount(period_idx, weights=severities, minlength=n_simulations)

        return {'aggregate_losses': aggregate, 'claim_counts': counts}
