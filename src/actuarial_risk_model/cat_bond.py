"""
Excess-of-loss catastrophe bond pricing for flood risk, driven by Extreme
Value Theory fitted to real daily rainfall exceedances (reuses
extreme_value.ExtremeValueModel's GPD fit). River gauge / streamflow data for
Kenyan rivers isn't freely available via a public API, so peak daily rainfall
in the lower Tana River basin (Garissa) is used as a public proxy for flood
severity -- the pricing mechanics below are agnostic to which index drives
the trigger.

Ships with real daily rainfall for Garissa, 2001-2023 (NASA POWER); see
data/climate/_convert.py for provenance.
"""
from pathlib import Path
from typing import Dict, List, Optional, Union
import csv
import numpy as np
from scipy.stats import genpareto

from .extreme_value import ExtremeValueModel

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "climate" / "rainfall_daily_garissa.csv"


def load_daily_rainfall() -> List[float]:
    """Real daily rainfall (mm) for Garissa, lower Tana River basin, 2001-2023."""
    with DATA_PATH.open() as f:
        return [float(r['rainfall_mm']) for r in csv.DictReader(f)]


def flood_payout_fraction(rainfall_mm: Union[float, np.ndarray], attachment_mm: float,
                           exhaustion_mm: float) -> Union[float, np.ndarray]:
    """
    Fraction of principal lost for an excess-rainfall (flood) trigger: 0 at
    or below `attachment_mm`, scaling linearly to 1 at or above `exhaustion_mm`.
    """
    if exhaustion_mm <= attachment_mm:
        raise ValueError("exhaustion_mm must exceed attachment_mm for an excess-rainfall trigger")
    scalar_input = np.isscalar(rainfall_mm)
    frac = (np.asarray(rainfall_mm, dtype=float) - attachment_mm) / (exhaustion_mm - attachment_mm)
    clipped = np.clip(frac, 0.0, 1.0)
    return float(clipped) if scalar_input else clipped


class FloodCatBond:

    @staticmethod
    def simulate_annual_max_rainfall(shape: float, scale: float, threshold: float,
                                      exceedance_rate: float, n_years: int,
                                      seed: Optional[int] = None) -> np.ndarray:
        """
        Simulate `n_years` of annual maximum daily rainfall from the fitted
        GPD tail beyond `threshold`.

        Number of threshold-exceedance days per year ~ Poisson(exceedance_rate
        x 365.25); this treats exceedance days as independent, which
        understates clustering during a single multi-day storm system. Years
        with zero exceedance days get `threshold` itself as a conservative
        floor for the annual max.
        """
        rng = np.random.default_rng(seed)
        n_exceed_days = rng.poisson(exceedance_rate * 365.25, n_years)
        annual_max = np.full(n_years, threshold, dtype=float)
        for i, n_days in enumerate(n_exceed_days):
            if n_days > 0:
                excess = genpareto.rvs(shape, scale=scale, size=n_days, random_state=rng)
                annual_max[i] = threshold + excess.max()
        return annual_max

    @staticmethod
    def price_bond(annual_max_rainfall: np.ndarray, attachment_mm: float, exhaustion_mm: float,
                    principal: float, risk_multiple: float = 3.0) -> Dict[str, float]:
        """
        Price a flood cat bond from a simulated annual-max-rainfall
        distribution.

        risk_multiple: ratio of coupon spread to modeled expected loss that
        investors demand for bearing the risk -- cat bonds have historically
        priced at roughly 2-5x modeled expected loss; 3x is a representative
        middle used here as a default.
        """
        payout_fractions = flood_payout_fraction(annual_max_rainfall, attachment_mm, exhaustion_mm)
        expected_loss_pct = float(np.mean(payout_fractions))
        return {
            'expected_loss_pct': expected_loss_pct,
            'expected_loss_amount': expected_loss_pct * principal,
            'probability_of_attachment': float(np.mean(payout_fractions > 0)),
            'probability_of_exhaustion': float(np.mean(payout_fractions >= 1)),
            'coupon_spread_pct': expected_loss_pct * risk_multiple,
            'annual_coupon_amount': expected_loss_pct * risk_multiple * principal,
        }
