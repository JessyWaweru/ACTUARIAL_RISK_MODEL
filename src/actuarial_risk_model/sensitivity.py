"""
Generic sensitivity / stress-test sweep utilities.

These take a `compute_fn(params) -> float` closure so callers (e.g. API
routers) can sweep any underlying calculation -- premium, VaR, reserves,
diversification benefit -- without this module needing to know about them.
"""
from typing import Any, Callable, Dict, List
import numpy as np


def one_way_sensitivity(base_params: Dict[str, float],
                         param_name: str,
                         values: List[float],
                         compute_fn: Callable[[Dict[str, float]], float]) -> Dict[str, Any]:
    """Vary a single parameter across `values`, holding everything else fixed."""
    results = []
    for v in values:
        params = {**base_params, param_name: v}
        results.append({'value': v, 'output': compute_fn(params)})

    base_output = compute_fn(base_params)
    return {
        'parameter': param_name,
        'base_value': base_params.get(param_name),
        'base_output': base_output,
        'results': results,
    }


def two_way_sensitivity(base_params: Dict[str, float],
                         param_x: str,
                         values_x: List[float],
                         param_y: str,
                         values_y: List[float],
                         compute_fn: Callable[[Dict[str, float]], float]) -> Dict[str, Any]:
    """Vary two parameters over a grid, holding everything else fixed."""
    grid = []
    for vx in values_x:
        row = []
        for vy in values_y:
            params = {**base_params, param_x: vx, param_y: vy}
            row.append(compute_fn(params))
        grid.append(row)

    return {
        'param_x': param_x,
        'values_x': values_x,
        'param_y': param_y,
        'values_y': values_y,
        'grid': grid,
    }
