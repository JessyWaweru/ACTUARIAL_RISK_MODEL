import numpy as np
from fastapi import APIRouter, HTTPException

from ...area_yield import AreaYieldInsurance, load_yield_series
from ...risk_model import RiskModel
from ..schemas import AreaYieldRequest, AreaYieldResponse
from ..utils import build_histogram

router = APIRouter(prefix="/api/area-yield", tags=["area-yield"])


@router.post("/analyze", response_model=AreaYieldResponse)
def analyze(req: AreaYieldRequest) -> AreaYieldResponse:
    yield_by_year = load_yield_series()
    trend = AreaYieldInsurance.fit_trend(yield_by_year)

    try:
        indemnities = AreaYieldInsurance.historical_indemnities(
            yield_by_year, trend, req.coverage_level, req.price_per_kg
        )
        pricing = AreaYieldInsurance.premium_from_indemnities(
            np.array(list(indemnities.values())), req.risk_load, req.expense_load
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    years = sorted(yield_by_year)

    # Simulate future indemnities: trend yield in a representative future year,
    # perturbed by the historical (detrended) residual distribution.
    model = RiskModel(seed=req.seed)
    future_year = years[-1] + 1
    trend_at_future = AreaYieldInsurance.trend_yield(trend, future_year)
    residuals = model.monte_carlo_simulation(
        'normal', {'mean': 0.0, 'std_dev': trend['residual_std']}, simulations=req.simulations
    )
    simulated_actual_yield = np.maximum(trend_at_future + residuals, 0.0)
    guaranteed = req.coverage_level * trend_at_future
    simulated_indemnities = np.maximum(guaranteed - simulated_actual_yield, 0.0) * req.price_per_kg

    return AreaYieldResponse(
        years=years,
        actual_yield=[yield_by_year[y] for y in years],
        trend_yield=[AreaYieldInsurance.trend_yield(trend, y) for y in years],
        historical_indemnities=[indemnities[y] for y in years],
        trend_slope=trend['slope'],
        residual_std=trend['residual_std'],
        pure_premium=pricing['pure_premium'],
        risk_load=pricing['risk_load'],
        gross_premium=pricing['gross_premium'],
        loss_ratio=pricing['loss_ratio'],
        var=float(model.calculate_var(simulated_indemnities, req.confidence)),
        tvar=float(model.calculate_tvar(simulated_indemnities, req.confidence)),
        histogram=build_histogram(simulated_indemnities),
    )
