"""
Area-yield index crop insurance: payout is triggered when a region's average
yield -- not an individual farmer's -- falls below a guaranteed fraction of
its trend yield. This avoids per-farmer loss adjustment and is the mechanism
real large-scale schemes use (e.g. ACRE Africa, African Risk Capacity).

Ships with Kenya's real national cereal yield series (World Bank Open Data,
kg/hectare, 1961-2023) as the area index; see data/agriculture/. County-level
yield isn't available via a free public API, so the national series stands in
for a county's area index here -- the methodology is identical either way.
"""
from pathlib import Path
from typing import Dict
import csv
import numpy as np

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "agriculture" / "kenya_cereal_yield.csv"


def load_yield_series() -> Dict[int, float]:
    """Real Kenya national cereal yield (kg/hectare) by year, from World Bank Open Data."""
    with DATA_PATH.open() as f:
        return {int(r['year']): float(r['yield_kg_per_ha']) for r in csv.DictReader(f)}


class AreaYieldInsurance:

    @staticmethod
    def fit_trend(yield_by_year: Dict[int, float]) -> Dict[str, float]:
        """
        OLS trend line through the yield series, capturing the technology-
        driven yield improvement over time so payouts reflect a genuine
        shortfall rather than the historical low yields of decades ago.
        """
        years = np.array(sorted(yield_by_year))
        values = np.array([yield_by_year[y] for y in years])
        slope, intercept = np.polyfit(years, values, 1)
        residuals = values - (slope * years + intercept)
        return {'slope': float(slope), 'intercept': float(intercept), 'residual_std': float(residuals.std(ddof=1))}

    @staticmethod
    def trend_yield(trend: Dict[str, float], year: int) -> float:
        return trend['slope'] * year + trend['intercept']

    @staticmethod
    def indemnity(actual_yield: float, guaranteed_yield: float, price_per_kg: float) -> float:
        """Indemnity per hectare = shortfall below the guaranteed yield x price."""
        return max(guaranteed_yield - actual_yield, 0.0) * price_per_kg

    @staticmethod
    def historical_indemnities(yield_by_year: Dict[int, float], trend: Dict[str, float],
                                coverage_level: float, price_per_kg: float) -> Dict[int, float]:
        """coverage_level: fraction of trend yield guaranteed, e.g. 0.8 = 80%."""
        if not 0 < coverage_level <= 1:
            raise ValueError("coverage_level must be in (0, 1]")
        result = {}
        for year, actual in yield_by_year.items():
            guaranteed = coverage_level * AreaYieldInsurance.trend_yield(trend, year)
            result[year] = AreaYieldInsurance.indemnity(actual, guaranteed, price_per_kg)
        return result

    @staticmethod
    def premium_from_indemnities(indemnities: np.ndarray, risk_load: float = 0.2,
                                  expense_load: float = 0.15) -> Dict[str, float]:
        indemnities = np.asarray(indemnities, dtype=float)
        pure_premium = float(np.mean(indemnities))
        risk_load_amount = risk_load * float(np.std(indemnities))
        gross = (pure_premium + risk_load_amount) * (1 + expense_load)
        return {
            'pure_premium': pure_premium,
            'risk_load': risk_load_amount,
            'gross_premium': gross,
            'loss_ratio': pure_premium / (pure_premium + risk_load_amount) if (pure_premium + risk_load_amount) else float('nan'),
        }
