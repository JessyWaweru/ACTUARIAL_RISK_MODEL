from fastapi import APIRouter, HTTPException

from ...risk_model import RiskModel
from ...sensitivity import one_way_sensitivity
from ..schemas import (
    PremiumSensitivityRequest,
    SensitivityPoint,
    SensitivityResponse,
    VarSensitivityRequest,
)

router = APIRouter(prefix="/api/sensitivity", tags=["sensitivity"])


@router.post("/premium", response_model=SensitivityResponse)
def premium_sensitivity(req: PremiumSensitivityRequest) -> SensitivityResponse:
    model = RiskModel()

    def compute(params: dict) -> float:
        return model.calculate_premium(
            exposure=params['exposure'],
            frequency=params['frequency'],
            severity=params['severity'],
            risk_load=params['risk_load'],
            expense_load=params['expense_load'],
        )

    try:
        result = one_way_sensitivity(req.base.model_dump(), req.param_name, req.values, compute)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return SensitivityResponse(
        parameter=result['parameter'],
        base_value=result['base_value'],
        base_output=result['base_output'],
        results=[SensitivityPoint(**r) for r in result['results']],
    )


@router.post("/var", response_model=SensitivityResponse)
def var_sensitivity(req: VarSensitivityRequest) -> SensitivityResponse:
    def compute(params: dict) -> float:
        model = RiskModel(seed=params.get('seed'))
        sim_params = {'mean': params['mean']}
        if params.get('dist') in ('normal', 'lognormal', 'gamma'):
            sim_params['std_dev'] = params['std_dev']
        losses = model.monte_carlo_simulation(params['dist'], sim_params, simulations=params['simulations'])
        return float(model.calculate_var(losses, req.confidence))

    try:
        result = one_way_sensitivity(req.base.model_dump(), req.param_name, req.values, compute)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return SensitivityResponse(
        parameter=result['parameter'],
        base_value=result['base_value'],
        base_output=result['base_output'],
        results=[SensitivityPoint(**r) for r in result['results']],
    )
