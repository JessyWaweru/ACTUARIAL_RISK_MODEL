"""
Excess-of-loss reinsurance layer pricing with reinstatements, and Rate on Line.

Convention used (a common simplification): reinstatement premium is charged
pro-rata as-to-amount (proportional to the limit reinstated) and 100% as-to-time
(no proration for time remaining in the period), at `reinstatement_cost_pct` of
the layer's base rate on line.
"""
from typing import Dict, List, Sequence
import numpy as np


def rate_on_line(premium: float, limit: float) -> float:
    """RoL = premium / limit."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    return premium / limit


def price_xol_with_reinstatements(annual_occurrence_losses: Sequence[Sequence[float]],
                                   attachment: float,
                                   limit: float,
                                   num_reinstatements: int,
                                   reinstatement_cost_pct: float = 1.0,
                                   risk_load_factor: float = 0.2) -> Dict[str, float]:
    """
    Price an excess-of-loss layer with reinstatements.

    Args:
        annual_occurrence_losses: one array of individual occurrence losses
            per simulated year (ragged, e.g. list of np.ndarray).
        attachment: layer attachment point.
        limit: layer limit (per occurrence and per reinstatement).
        num_reinstatements: number of reinstatements purchased.
        reinstatement_cost_pct: fraction of the base rate on line charged per
            unit of limit reinstated (1.0 = full pro-rata reinstatement cost).
        risk_load_factor: loading applied to the standard deviation of the
            first layer's annual recoveries.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    if num_reinstatements < 0:
        raise ValueError("num_reinstatements cannot be negative")

    aggregate_capacity = limit * (1 + num_reinstatements)
    first_layer_recovery = np.zeros(len(annual_occurrence_losses))
    total_recovery = np.zeros(len(annual_occurrence_losses))
    reinstatements_used = np.zeros(len(annual_occurrence_losses))

    for year_idx, occ_losses in enumerate(annual_occurrence_losses):
        remaining = aggregate_capacity
        year_recovery = 0.0
        for occ_loss in occ_losses:
            occ_layer = min(max(occ_loss - attachment, 0.0), limit)
            recoverable = min(occ_layer, remaining)
            year_recovery += recoverable
            remaining -= recoverable
            if remaining <= 0:
                break
        total_recovery[year_idx] = year_recovery
        first_layer_recovery[year_idx] = min(year_recovery, limit)
        reinstated_amount = max(year_recovery - limit, 0.0)
        reinstatements_used[year_idx] = min(reinstated_amount / limit, num_reinstatements)

    first_layer_pure_premium = float(np.mean(first_layer_recovery))
    risk_load = risk_load_factor * float(np.std(first_layer_recovery))
    first_layer_premium = first_layer_pure_premium + risk_load
    base_rol = rate_on_line(first_layer_premium, limit)

    expected_reinstated_limit = float(np.mean(np.minimum(reinstatements_used, num_reinstatements))) * limit
    reinstatement_premium = base_rol * reinstatement_cost_pct * expected_reinstated_limit

    gross_premium = first_layer_premium + reinstatement_premium
    expected_total_recovery = float(np.mean(total_recovery))

    return {
        'first_layer_pure_premium': first_layer_pure_premium,
        'risk_load': risk_load,
        'first_layer_premium': first_layer_premium,
        'rate_on_line': base_rol,
        'expected_reinstatements_used': float(np.mean(reinstatements_used)),
        'reinstatement_premium': reinstatement_premium,
        'gross_premium': gross_premium,
        'effective_rate_on_line': rate_on_line(gross_premium, limit),
        'expected_total_recovery': expected_total_recovery,
        'loss_ratio': expected_total_recovery / gross_premium if gross_premium > 0 else float('nan'),
    }
