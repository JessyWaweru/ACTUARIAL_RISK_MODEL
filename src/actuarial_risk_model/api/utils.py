from typing import Dict

import numpy as np
from fastapi import HTTPException

from ..risk_model import RiskModel
from .schemas import SimulationInput


def resolve_losses(sim: SimulationInput) -> np.ndarray:
    """Return raw losses if provided, else regenerate them from distribution params."""
    if sim.raw_losses is not None:
        if len(sim.raw_losses) == 0:
            raise HTTPException(status_code=400, detail="raw_losses cannot be empty")
        return np.asarray(sim.raw_losses, dtype=float)

    if sim.dist is None or sim.mean is None:
        raise HTTPException(status_code=400, detail="Provide either raw_losses, or dist + mean")

    model = RiskModel(seed=sim.seed)
    params: Dict[str, float] = {'mean': sim.mean}
    if sim.dist in ('normal', 'lognormal', 'gamma'):
        if sim.std_dev is None:
            raise HTTPException(status_code=400, detail=f"std_dev is required for {sim.dist} distribution")
        params['std_dev'] = sim.std_dev

    try:
        return model.monte_carlo_simulation(sim.dist, params, simulations=sim.simulations)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def build_histogram(losses: np.ndarray, bins: int = 50) -> Dict[str, list]:
    counts, edges = np.histogram(losses, bins=bins)
    return {'bin_edges': edges.tolist(), 'counts': counts.tolist()}


def build_loss_summary(losses: np.ndarray, bins: int = 50) -> Dict:
    percentiles = {f"p{p}": float(np.percentile(losses, p)) for p in (5, 25, 50, 75, 90, 95, 99)}
    return {
        'mean': float(np.mean(losses)),
        'std_dev': float(np.std(losses)),
        'min': float(np.min(losses)),
        'max': float(np.max(losses)),
        'percentiles': percentiles,
        'histogram': build_histogram(losses, bins=bins),
    }


def triangle_to_array(triangle: list) -> np.ndarray:
    return np.array([[np.nan if v is None else v for v in row] for row in triangle], dtype=float)
