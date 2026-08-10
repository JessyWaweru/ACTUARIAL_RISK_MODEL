"""
Extreme Value Theory: Peaks-Over-Threshold tail modeling with the Generalized
Pareto Distribution (GPD), for large-loss / catastrophe tail risk that a
single fitted body distribution (e.g. lognormal) tends to underestimate.
"""
from typing import Dict
import numpy as np
from scipy.stats import genpareto


class ExtremeValueModel:

    @staticmethod
    def fit_gpd(losses: np.ndarray, threshold: float) -> Dict[str, float]:
        """Fit a GPD to exceedances of `threshold` (peaks-over-threshold)."""
        losses = np.asarray(losses, dtype=float)
        exceedances = losses[losses > threshold] - threshold
        if len(exceedances) < 10:
            raise ValueError(
                "Need at least 10 exceedances above the threshold for a stable GPD fit; "
                f"got {len(exceedances)}. Try a lower threshold."
            )
        shape, _, scale = genpareto.fit(exceedances, floc=0)
        n = len(losses)
        n_exceed = len(exceedances)
        return {
            'shape': float(shape),
            'scale': float(scale),
            'threshold': float(threshold),
            'n_exceedances': n_exceed,
            'exceedance_rate': n_exceed / n,
        }

    @staticmethod
    def return_level(shape: float, scale: float, threshold: float, exceedance_rate: float,
                      return_period: float, events_per_period: float = 1.0) -> float:
        """Loss level expected to be exceeded once every `return_period` periods."""
        m = return_period * events_per_period
        if abs(shape) < 1e-8:
            return threshold + scale * np.log(m * exceedance_rate)
        return threshold + (scale / shape) * ((m * exceedance_rate) ** shape - 1)

    @staticmethod
    def tail_risk_metrics(shape: float, scale: float, threshold: float,
                           exceedance_rate: float, confidence: float) -> Dict[str, float]:
        """VaR and TVaR at `confidence`, extrapolated via the fitted GPD tail."""
        p_exceed = 1 - confidence
        if p_exceed >= exceedance_rate:
            raise ValueError(
                "confidence is too low for GPD extrapolation (falls within the empirical "
                "body, below the fitted threshold) -- use the empirical VaR instead"
            )
        ratio = p_exceed / exceedance_rate
        if abs(shape) < 1e-8:
            var = threshold - scale * np.log(ratio)
        else:
            var = threshold + (scale / shape) * (ratio ** (-shape) - 1)

        if shape < 1:
            tvar = (var + scale - shape * threshold) / (1 - shape)
        else:
            tvar = float('inf')

        return {'var': float(var), 'tvar': float(tvar)}
