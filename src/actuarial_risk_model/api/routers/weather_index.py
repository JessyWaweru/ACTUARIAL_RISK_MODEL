import numpy as np
from fastapi import APIRouter, HTTPException

from ...risk_model import RiskModel
from ...weather_index import WeatherIndexInsurance, load_monthly_rainfall
from ..schemas import WeatherIndexRequest, WeatherIndexResponse
from ..utils import build_histogram

router = APIRouter(prefix="/api/weather-index", tags=["weather-index"])


@router.post("/analyze", response_model=WeatherIndexResponse)
def analyze(req: WeatherIndexRequest) -> WeatherIndexResponse:
    monthly = load_monthly_rainfall(req.county)
    index_by_year = WeatherIndexInsurance.seasonal_index(monthly, req.trigger_months)
    if len(index_by_year) < 5:
        raise HTTPException(status_code=400, detail="Not enough complete seasons in the historical record")

    try:
        historical_payouts = WeatherIndexInsurance.historical_payouts(
            index_by_year, req.strike_mm, req.exit_mm, req.sum_insured
        )
        burn_cost = WeatherIndexInsurance.burn_cost_premium(
            np.array(list(historical_payouts.values())), req.risk_load, req.expense_load
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    index_values = np.array(list(index_by_year.values()))
    fitted_mean, fitted_std = float(index_values.mean()), float(index_values.std())

    model = RiskModel(seed=req.seed)
    simulated_index = model.monte_carlo_simulation(
        'gamma', {'mean': fitted_mean, 'std_dev': fitted_std}, simulations=req.simulations
    )
    simulated_payouts = WeatherIndexInsurance.payout_fraction(simulated_index, req.strike_mm, req.exit_mm) * req.sum_insured

    years = sorted(index_by_year)
    return WeatherIndexResponse(
        county=req.county,
        years=years,
        historical_index_mm=[index_by_year[y] for y in years],
        historical_payouts=[historical_payouts[y] for y in years],
        fitted_mean_mm=fitted_mean,
        fitted_std_mm=fitted_std,
        pure_premium=burn_cost['pure_premium'],
        risk_load=burn_cost['risk_load'],
        gross_premium=burn_cost['gross_premium'],
        loss_ratio=burn_cost['loss_ratio'],
        payout_probability=float(np.mean(simulated_payouts > 0)),
        simulated_mean_payout=float(np.mean(simulated_payouts)),
        var=float(model.calculate_var(simulated_payouts, req.confidence)),
        tvar=float(model.calculate_tvar(simulated_payouts, req.confidence)),
        histogram=build_histogram(simulated_payouts),
    )
