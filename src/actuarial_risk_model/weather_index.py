"""
Weather-index (parametric) insurance: payout is a function of a rainfall
index crossing a strike level, not an individually assessed claim -- the
standard mechanism for smallholder drought/livestock cover, where per-farmer
loss adjustment across a whole county is infeasible.

Ships with real historical monthly rainfall for three Kenyan counties
(Homa Bay, West Pokot, Turkana), sourced from NASA POWER (see
data/climate/_convert.py for provenance and re-fetch instructions).
"""
from pathlib import Path
from typing import Dict, List, Sequence, Union
import csv
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "climate"

COUNTIES: Dict[str, str] = {
    "homabay": "rainfall_monthly_homabay.csv",
    "westpokot": "rainfall_monthly_westpokot.csv",
    "turkana": "rainfall_monthly_turkana.csv",
}


def load_monthly_rainfall(county: str) -> List[Dict[str, float]]:
    """Real historical monthly rainfall (mm) for `county`, one of COUNTIES."""
    if county not in COUNTIES:
        raise ValueError(f"Unknown county '{county}'; choose from {sorted(COUNTIES)}")
    rows = []
    with (DATA_DIR / COUNTIES[county]).open() as f:
        for r in csv.DictReader(f):
            rows.append({'year': int(r['year']), 'month': int(r['month']), 'rainfall_mm': float(r['rainfall_mm'])})
    return rows


class WeatherIndexInsurance:
    """Rainfall index insurance with a drought (deficit-rainfall) trigger."""

    @staticmethod
    def seasonal_index(monthly_rainfall: Sequence[Dict], months: Sequence[int]) -> Dict[int, float]:
        """
        Cumulative rainfall over `months` (1-12) for each year present in
        `monthly_rainfall`. A year is only included if all requested months
        are present, so a partial final year doesn't understate its total.
        """
        by_year: Dict[int, Dict[int, float]] = {}
        for row in monthly_rainfall:
            by_year.setdefault(row['year'], {})[row['month']] = row['rainfall_mm']
        return {
            year: sum(months_map[m] for m in months)
            for year, months_map in by_year.items()
            if all(m in months_map for m in months)
        }

    @staticmethod
    def payout_fraction(index_value: Union[float, np.ndarray], strike: float,
                         exit_level: float) -> Union[float, np.ndarray]:
        """
        Fraction of sum insured paid out for a drought trigger: 0 at or above
        `strike`, scaling linearly to 1 at or below `exit_level`.
        """
        if exit_level >= strike:
            raise ValueError("exit_level must be below strike for a drought (rainfall-deficit) trigger")
        scalar_input = np.isscalar(index_value)
        frac = (strike - np.asarray(index_value, dtype=float)) / (strike - exit_level)
        clipped = np.clip(frac, 0.0, 1.0)
        return float(clipped) if scalar_input else clipped

    @staticmethod
    def historical_payouts(index_by_year: Dict[int, float], strike: float, exit_level: float,
                            sum_insured: float) -> Dict[int, float]:
        return {
            year: float(WeatherIndexInsurance.payout_fraction(value, strike, exit_level)) * sum_insured
            for year, value in index_by_year.items()
        }

    @staticmethod
    def burn_cost_premium(payouts: np.ndarray, risk_load: float = 0.2,
                           expense_load: float = 0.15) -> Dict[str, float]:
        """
        Burn-cost pricing: pure premium is the historical average payout,
        loaded for risk (proportional to payout volatility) and expenses --
        the standard first-pass pricing method for index insurance.
        """
        payouts = np.asarray(payouts, dtype=float)
        pure_premium = float(np.mean(payouts))
        risk_load_amount = risk_load * float(np.std(payouts))
        gross = (pure_premium + risk_load_amount) * (1 + expense_load)
        return {
            'pure_premium': pure_premium,
            'risk_load': risk_load_amount,
            'gross_premium': gross,
            'loss_ratio': pure_premium / (pure_premium + risk_load_amount) if (pure_premium + risk_load_amount) else float('nan'),
        }
