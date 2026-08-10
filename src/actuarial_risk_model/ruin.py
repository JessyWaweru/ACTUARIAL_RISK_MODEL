"""
Ruin theory: Cramer-Lundberg model of an insurer's surplus process.

Claims arrive as a Poisson process at rate `claim_rate`; premium accrues
continuously at rate c = claim_rate * mean_severity * (1 + premium_loading).
The adjustment coefficient R and Lundberg's inequality give an analytic bound
on ultimate ruin probability for light-tailed severities; a Monte Carlo
simulation gives a finite-horizon estimate for any severity distribution.
"""
from typing import Dict, Optional
import numpy as np
from scipy.optimize import brentq


class RuinTheory:

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def adjustment_coefficient(claim_rate: float,
                                severity_dist: str,
                                severity_params: Dict[str, float],
                                premium_loading: float) -> float:
        """
        Solve the Lundberg equation for the adjustment coefficient R.
        Requires a light-tailed severity (moment generating function exists).
        """
        mean = severity_params['mean']
        c = claim_rate * mean * (1 + premium_loading)

        if severity_dist == 'exponential':
            beta = 1 / mean
            upper = beta * 0.999999

            def lundberg_eq(r: float) -> float:
                mgf = beta / (beta - r)
                return claim_rate + c * r - claim_rate * mgf
        elif severity_dist == 'gamma':
            std = severity_params['std_dev']
            shape = mean ** 2 / std ** 2
            rate = mean / std ** 2
            upper = rate * 0.999999

            def lundberg_eq(r: float) -> float:
                mgf = (rate / (rate - r)) ** shape
                return claim_rate + c * r - claim_rate * mgf
        else:
            raise ValueError(
                "adjustment_coefficient supports 'exponential' or 'gamma' severities only "
                "(the Lundberg equation requires a light-tailed severity distribution)"
            )

        return float(brentq(lundberg_eq, 1e-10, upper))

    @staticmethod
    def ruin_probability_bound(initial_surplus: float, adjustment_coefficient: float) -> float:
        """Lundberg's inequality: psi(u) <= exp(-R*u). Valid for any light-tailed severity."""
        return float(np.exp(-adjustment_coefficient * initial_surplus))

    @staticmethod
    def ruin_probability_exact_exponential(initial_surplus: float,
                                            claim_rate: float,
                                            mean_severity: float,
                                            premium_loading: float) -> float:
        """Exact ultimate ruin probability for exponential claim sizes."""
        if premium_loading <= 0:
            raise ValueError("premium_loading must be > 0 for ruin probability to be < 1")
        c = claim_rate * mean_severity * (1 + premium_loading)
        r = premium_loading / (mean_severity * (1 + premium_loading))
        rho = claim_rate * mean_severity / c
        return float(rho * np.exp(-r * initial_surplus))

    def _draw_severity(self, dist: str, params: Dict[str, float], n: int) -> np.ndarray:
        mean = params['mean']
        if dist == 'exponential':
            return self.rng.exponential(mean, n)
        elif dist == 'gamma':
            std = params['std_dev']
            shape = mean ** 2 / std ** 2
            scale = std ** 2 / mean
            return self.rng.gamma(shape, scale, n)
        elif dist == 'lognormal':
            std = params['std_dev']
            sigma2 = np.log(1 + std ** 2 / mean ** 2)
            mu = np.log(mean) - sigma2 / 2
            return self.rng.lognormal(mu, np.sqrt(sigma2), n)
        else:
            raise ValueError(f"Unsupported severity distribution: {dist}")

    def simulate_finite_horizon_ruin(self,
                                      initial_surplus: float,
                                      claim_rate: float,
                                      severity_dist: str,
                                      severity_params: Dict[str, float],
                                      premium_loading: float,
                                      time_horizon: float,
                                      n_paths: int = 5_000) -> Dict[str, float]:
        """Monte Carlo estimate of P(ruin within `time_horizon`) for any severity."""
        mean = severity_params['mean']
        c = claim_rate * mean * (1 + premium_loading)

        ruin_count = 0
        ruin_times = []
        for _ in range(n_paths):
            n_claims = self.rng.poisson(claim_rate * time_horizon)
            if n_claims == 0:
                continue
            arrival_times = np.sort(self.rng.uniform(0, time_horizon, n_claims))
            claims = self._draw_severity(severity_dist, severity_params, n_claims)
            surplus_path = initial_surplus + c * arrival_times - np.cumsum(claims)
            ruin_idx = np.argmax(surplus_path < 0) if np.any(surplus_path < 0) else -1
            if ruin_idx != -1:
                ruin_count += 1
                ruin_times.append(arrival_times[ruin_idx])

        ruin_probability = ruin_count / n_paths
        return {
            'ruin_probability': ruin_probability,
            'n_paths': n_paths,
            'n_ruined': ruin_count,
            'mean_time_to_ruin': float(np.mean(ruin_times)) if ruin_times else None,
        }
