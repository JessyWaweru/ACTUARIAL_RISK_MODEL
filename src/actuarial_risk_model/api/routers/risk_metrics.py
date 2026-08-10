from fastapi import APIRouter, HTTPException

from ...risk_model import RiskModel
from ..schemas import RiskMetricsRequest, RiskMetricsResponse
from ..utils import resolve_losses

router = APIRouter(prefix="/api/risk-metrics", tags=["risk-metrics"])


@router.post("", response_model=RiskMetricsResponse)
def calculate_risk_metrics(req: RiskMetricsRequest) -> RiskMetricsResponse:
    losses = resolve_losses(req.simulation)
    model = RiskModel()
    try:
        var = model.calculate_var(losses, req.confidence)
        tvar = model.calculate_tvar(losses, req.confidence)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return RiskMetricsResponse(
        mean=float(losses.mean()),
        std_dev=float(losses.std()),
        var=float(var),
        tvar=float(tvar),
        max_loss=float(losses.max()),
        min_loss=float(losses.min()),
        confidence=req.confidence,
    )
