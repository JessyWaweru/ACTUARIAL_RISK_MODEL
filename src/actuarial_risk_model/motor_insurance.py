"""
Motor insurance pricing for the Kenyan market: frequency/severity by vehicle
class, plus a bonus-malus (no-claims discount) scale.

Vehicle-class claim frequency/severity figures are illustrative -- calibrated
to well-documented industry patterns (PSV/matatu and commercial vehicles
carry materially higher claim frequency than private cars; motorcycles/boda
boda highest of all) rather than pulled from a specific insurer's book, since
granular Kenyan motor claims data isn't publicly available.
"""
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from .aggregate_loss import AggregateLossModel


@dataclass(frozen=True)
class VehicleClass:
    label: str
    annual_frequency: float  # claims per vehicle-year
    mean_severity: float     # KES
    severity_std: float      # KES


VEHICLE_CLASSES: Dict[str, VehicleClass] = {
    'private': VehicleClass('Private car', annual_frequency=0.08, mean_severity=120_000, severity_std=90_000),
    'psv': VehicleClass('PSV / matatu', annual_frequency=0.35, mean_severity=180_000, severity_std=150_000),
    'commercial': VehicleClass('Commercial / haulage', annual_frequency=0.22, mean_severity=350_000, severity_std=300_000),
    'motorcycle': VehicleClass('Motorcycle / boda boda', annual_frequency=0.45, mean_severity=60_000, severity_std=50_000),
}

# Discount off the base premium by consecutive claim-free years (capped at 5+).
BONUS_MALUS_SCALE: Dict[int, float] = {0: 0.0, 1: 0.10, 2: 0.20, 3: 0.30, 4: 0.40, 5: 0.50}


def _get_vehicle_class(vehicle_class: str) -> VehicleClass:
    if vehicle_class not in VEHICLE_CLASSES:
        raise ValueError(f"Unknown vehicle_class '{vehicle_class}'; choose from {sorted(VEHICLE_CLASSES)}")
    return VEHICLE_CLASSES[vehicle_class]


class MotorInsurancePricing:

    @staticmethod
    def base_premium(vehicle_class: str, risk_load: float = 0.25, expense_load: float = 0.2) -> Dict[str, float]:
        vc = _get_vehicle_class(vehicle_class)
        pure_premium = vc.annual_frequency * vc.mean_severity
        risk_load_amount = risk_load * pure_premium
        gross = (pure_premium + risk_load_amount) * (1 + expense_load)
        return {
            'vehicle_class': vc.label,
            'pure_premium': pure_premium,
            'risk_load': risk_load_amount,
            'gross_premium': gross,
        }

    @staticmethod
    def bonus_malus_premium(base_gross_premium: float, claim_free_years: int) -> Dict[str, float]:
        capped = max(0, min(claim_free_years, max(BONUS_MALUS_SCALE)))
        discount = BONUS_MALUS_SCALE[capped]
        return {
            'claim_free_years': claim_free_years,
            'discount_pct': discount,
            'adjusted_premium': base_gross_premium * (1 - discount),
        }

    @staticmethod
    def simulate_fleet_losses(vehicle_class: str, n_vehicles: int, n_years: int = 10_000,
                               seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Simulate a fleet of `n_vehicles` over `n_years` independent policy
        years via the compound Poisson-lognormal aggregate loss model.
        """
        vc = _get_vehicle_class(vehicle_class)
        model = AggregateLossModel(seed=seed)
        return model.simulate(
            freq_dist='poisson',
            freq_params={'mean': vc.annual_frequency * n_vehicles},
            sev_dist='lognormal',
            sev_params={'mean': vc.mean_severity, 'std_dev': vc.severity_std},
            n_simulations=n_years,
        )
