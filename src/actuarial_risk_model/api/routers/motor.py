from fastapi import APIRouter, HTTPException

from ...motor_insurance import MotorInsurancePricing
from ...risk_model import RiskModel
from ..schemas import (
    MotorFleetSimulationRequest, MotorFleetSimulationResponse,
    MotorPremiumRequest, MotorPremiumResponse,
)
from ..utils import build_histogram

router = APIRouter(prefix="/api/motor", tags=["motor"])


@router.post("/premium", response_model=MotorPremiumResponse)
def premium(req: MotorPremiumRequest) -> MotorPremiumResponse:
    try:
        base = MotorInsurancePricing.base_premium(req.vehicle_class, req.risk_load, req.expense_load)
        bonus_malus = MotorInsurancePricing.bonus_malus_premium(base['gross_premium'], req.claim_free_years)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return MotorPremiumResponse(
        vehicle_class=base['vehicle_class'],
        pure_premium=base['pure_premium'],
        risk_load=base['risk_load'],
        gross_premium=base['gross_premium'],
        discount_pct=bonus_malus['discount_pct'],
        adjusted_premium=bonus_malus['adjusted_premium'],
    )


@router.post("/fleet-simulation", response_model=MotorFleetSimulationResponse)
def fleet_simulation(req: MotorFleetSimulationRequest) -> MotorFleetSimulationResponse:
    try:
        result = MotorInsurancePricing.simulate_fleet_losses(
            req.vehicle_class, req.n_vehicles, req.n_years, req.seed
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    losses = result['aggregate_losses']
    model = RiskModel()
    return MotorFleetSimulationResponse(
        mean_annual_claims=float(losses.mean()),
        var=float(model.calculate_var(losses, req.confidence)),
        tvar=float(model.calculate_tvar(losses, req.confidence)),
        histogram=build_histogram(losses),
    )
