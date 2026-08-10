"""
Correlated multi-line portfolio aggregation via a Gaussian copula.

Simulating each line independently and summing overstates capital needs when
lines aren't perfectly correlated; this joins per-line marginals through a
Gaussian copula so the simulated total reflects real diversification benefit.
"""
from typing import Dict, List, Optional
import numpy as np
from scipy.stats import norm, lognorm, gamma


class CorrelatedPortfolio:

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def simulate(self,
                 lines: List[Dict[str, float]],
                 correlation_matrix: np.ndarray,
                 n_simulations: int = 10_000) -> Dict[str, np.ndarray]:
        """
        Args:
            lines: list of {'name': str, 'dist': 'lognormal'|'gamma', 'mean': float, 'std_dev': float}
            correlation_matrix: k x k Pearson correlation matrix between lines.
        """
        k = len(lines)
        correlation_matrix = np.asarray(correlation_matrix, dtype=float)
        if correlation_matrix.shape != (k, k):
            raise ValueError(f"correlation_matrix must be {k}x{k} for {k} lines")

        eigvals, eigvecs = np.linalg.eigh(correlation_matrix)
        if np.any(eigvals < -1e-8):
            raise ValueError("correlation_matrix is not positive semi-definite")
        transform = eigvecs @ np.diag(np.sqrt(np.clip(eigvals, 0, None)))

        z = self.rng.standard_normal((n_simulations, k)) @ transform.T
        u = norm.cdf(z)

        losses = np.zeros((n_simulations, k))
        for j, line in enumerate(lines):
            dist, mean, std = line['dist'], line['mean'], line['std_dev']
            if dist == 'lognormal':
                sigma2 = np.log(1 + std ** 2 / mean ** 2)
                mu = np.log(mean) - sigma2 / 2
                losses[:, j] = lognorm.ppf(u[:, j], s=np.sqrt(sigma2), scale=np.exp(mu))
            elif dist == 'gamma':
                shape = mean ** 2 / std ** 2
                scale = std ** 2 / mean
                losses[:, j] = gamma.ppf(u[:, j], a=shape, scale=scale)
            else:
                raise ValueError(f"Unsupported distribution: {dist}")

        total = losses.sum(axis=1)
        per_line_std = np.std(losses, axis=0)
        sum_of_std = float(np.sum(per_line_std))
        correlated_std = float(np.std(total))
        diversification_benefit_pct = (1 - correlated_std / sum_of_std) * 100 if sum_of_std > 0 else 0.0

        return {
            'total_losses': total,
            'per_line_losses': losses,
            'line_names': [line['name'] for line in lines],
            'correlated_std': correlated_std,
            'sum_of_individual_std': sum_of_std,
            'diversification_benefit_pct': diversification_benefit_pct,
        }
