from fastapi import APIRouter, HTTPException

from ...reinsurance_advanced import price_xol_with_reinstatements, rate_on_line
from ...risk_model import RiskModel
from ..schemas import (
    RateOnLineRequest,
    RateOnLineResponse,
    ReinsuranceLayerRequest,
    ReinsuranceLayerResponse,
    XolReinstatementRequest,
    XolReinstatementResponse,
)
from ..utils import resolve_losses

router = APIRouter(prefix="/api/reinsurance", tags=["reinsurance"])


@router.post("/layer", response_model=ReinsuranceLayerResponse)
def price_layer(req: ReinsuranceLayerRequest) -> ReinsuranceLayerResponse:
    losses = resolve_losses(req.simulation)
    model = RiskModel()
    result = model.price_reinsurance_layer(losses, req.attachment, req.limit)
    return ReinsuranceLayerResponse(**{k: float(v) for k, v in result.items()})


@router.post("/rate-on-line", response_model=RateOnLineResponse)
def compute_rate_on_line(req: RateOnLineRequest) -> RateOnLineResponse:
    return RateOnLineResponse(rate_on_line=rate_on_line(req.premium, req.limit))


@router.post("/xol-reinstatements", response_model=XolReinstatementResponse)
def price_xol(req: XolReinstatementRequest) -> XolReinstatementResponse:
    try:
        result = price_xol_with_reinstatements(
            req.annual_occurrence_losses,
            req.attachment,
            req.limit,
            req.num_reinstatements,
            req.reinstatement_cost_pct,
            req.risk_load_factor,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return XolReinstatementResponse(**result)
