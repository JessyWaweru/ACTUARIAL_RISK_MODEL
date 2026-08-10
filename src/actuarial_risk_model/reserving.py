"""
Loss reserving methods beyond the deterministic chain ladder in risk_model.py:

- Mack's stochastic chain ladder: same age-to-age development factors, plus a
  standard error of the reserve estimate (process + parameter uncertainty).
- Bornhuetter-Ferguson: blends an a-priori expected loss ratio with chain
  ladder development, useful when early-year development data is sparse.

Triangles are n x n cumulative-claims arrays; unpopulated (future) cells must
be np.nan. Row i = accident/origin year, column k = development period.
"""
from typing import Dict
import numpy as np


def _dev_factors_and_sigma2(triangle: np.ndarray) -> Dict[str, np.ndarray]:
    n = triangle.shape[0]
    tri = triangle.astype(float)

    dev_factors = np.zeros(n - 1)
    sigma2 = np.zeros(n - 1)

    for k in range(n - 1):
        valid_rows = n - k - 1
        c_k = tri[:valid_rows, k]
        c_k1 = tri[:valid_rows, k + 1]
        dev_factors[k] = np.sum(c_k1) / np.sum(c_k)

        if valid_rows > 1:
            resid = c_k * (c_k1 / c_k - dev_factors[k]) ** 2
            sigma2[k] = np.sum(resid) / (valid_rows - 1)
        elif k >= 2:
            sigma2[k] = min(sigma2[k - 1], sigma2[k - 2])
        elif k >= 1:
            sigma2[k] = sigma2[k - 1]
        else:
            sigma2[k] = 0.0

    return {'dev_factors': dev_factors, 'sigma2': sigma2}


def _complete_triangle(triangle: np.ndarray, dev_factors: np.ndarray) -> np.ndarray:
    n = triangle.shape[0]
    completed = triangle.astype(float).copy()
    for i in range(1, n):
        for j in range(n - i, n):
            if np.isnan(completed[i, j]):
                completed[i, j] = completed[i, j - 1] * dev_factors[j - 1]
    return completed


def mack_chain_ladder(triangle: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Mack's stochastic chain ladder.

    Note: `total_standard_error` sums per-accident-year MSE and ignores the
    cross-accident-year covariance term from Mack's full formula -- a common
    simplifying approximation that slightly understates total uncertainty.
    """
    n = triangle.shape[0]
    tri = triangle.astype(float)
    params = _dev_factors_and_sigma2(tri)
    dev_factors, sigma2 = params['dev_factors'], params['sigma2']

    completed = _complete_triangle(tri, dev_factors)
    ultimate = completed[:, -1]
    latest_diagonal = np.array([tri[i, n - 1 - i] for i in range(n)])
    reserve_by_year = ultimate - latest_diagonal

    mse = np.zeros(n)
    for i in range(1, n):
        dev_age = n - 1 - i
        variance_sum = 0.0
        running = tri[i, dev_age]
        for k in range(dev_age, n - 1):
            column_sum = np.nansum(tri[:n - k - 1, k])
            variance_sum += (sigma2[k] / dev_factors[k] ** 2) * (1.0 / running + 1.0 / column_sum)
            running = running * dev_factors[k]
        mse[i] = (ultimate[i] ** 2) * variance_sum

    se_by_year = np.sqrt(mse)
    total_reserve = float(np.sum(reserve_by_year))
    total_se = float(np.sqrt(np.sum(mse)))

    return {
        'dev_factors': dev_factors,
        'ultimate_by_year': ultimate,
        'reserve_by_year': reserve_by_year,
        'standard_error_by_year': se_by_year,
        'total_reserve': total_reserve,
        'total_standard_error': total_se,
        'coefficient_of_variation': total_se / total_reserve if total_reserve else float('nan'),
    }


def bornhuetter_ferguson(triangle: np.ndarray,
                          expected_loss_ratio: float,
                          premium: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Bornhuetter-Ferguson reserving.

    Args:
        triangle: n x n cumulative claims triangle (future cells = np.nan).
        expected_loss_ratio: a-priori expected ultimate loss ratio.
        premium: earned premium per accident year (length n).
    """
    n = triangle.shape[0]
    tri = triangle.astype(float)
    premium = np.asarray(premium, dtype=float)
    if premium.shape != (n,):
        raise ValueError(f"premium must have length {n}")

    dev_factors = _dev_factors_and_sigma2(tri)['dev_factors']

    cum_to_ult = np.ones(n)
    running = 1.0
    for k in range(n - 2, -1, -1):
        running *= dev_factors[k]
        cum_to_ult[k] = running
    cum_to_ult[n - 1] = 1.0

    latest_diagonal = np.array([tri[i, n - 1 - i] for i in range(n)])
    expected_ultimate = premium * expected_loss_ratio

    pct_reported = np.zeros(n)
    reserves = np.zeros(n)
    ultimate = np.zeros(n)
    for i in range(n):
        dev_age = n - 1 - i
        factor_to_ult = cum_to_ult[dev_age]
        pct_rep = 1.0 / factor_to_ult
        pct_reported[i] = pct_rep
        reserves[i] = expected_ultimate[i] * (1 - pct_rep)
        ultimate[i] = latest_diagonal[i] + reserves[i]

    return {
        'dev_factors': dev_factors,
        'pct_reported_by_year': pct_reported,
        'reserve_by_year': reserves,
        'ultimate_by_year': ultimate,
        'total_reserve': float(np.sum(reserves)),
    }
