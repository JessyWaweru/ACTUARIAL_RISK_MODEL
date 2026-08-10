from fastapi import APIRouter, HTTPException

from ...extreme_value import ExtremeValueModel
from ..schemas import ExtremeValueRequest, ExtremeValueResponse, ReturnLevelPoint
from ..utils import resolve_losses

router = APIRouter(prefix="/api/extreme-value", tags=["extreme-value"])


@router.post("/analyze", response_model=ExtremeValueResponse)
def analyze(req: ExtremeValueRequest) -> ExtremeValueResponse:
    losses = resolve_losses(req.simulation)
    try:
        fit = ExtremeValueModel.fit_gpd(losses, req.threshold)
        return_levels = [
            ReturnLevelPoint(
                return_period=p,
                level=ExtremeValueModel.return_level(
                    fit['shape'], fit['scale'], fit['threshold'], fit['exceedance_rate'], p, req.events_per_period
                ),
            )
            for p in req.return_periods
        ]
        tail = ExtremeValueModel.tail_risk_metrics(
            fit['shape'], fit['scale'], fit['threshold'], fit['exceedance_rate'], req.confidence
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ExtremeValueResponse(
        shape=fit['shape'],
        scale=fit['scale'],
        threshold=fit['threshold'],
        n_exceedances=fit['n_exceedances'],
        exceedance_rate=fit['exceedance_rate'],
        return_levels=return_levels,
        tail_var=tail['var'],
        tail_tvar=tail['tvar'],
    )
