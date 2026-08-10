from fastapi import APIRouter, HTTPException

from ...risk_model import RiskModel
from ..schemas import PremiumRequest, PremiumResponse

router = APIRouter(prefix="/api/premium", tags=["premium"])


@router.post("/calculate", response_model=PremiumResponse)
def calculate_premium(req: PremiumRequest) -> PremiumResponse:
    model = RiskModel()
    try:
        gross = model.calculate_premium(
            exposure=req.exposure,
            frequency=req.frequency,
            severity=req.severity,
            risk_load=req.risk_load,
            expense_load=req.expense_load,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PremiumResponse(gross_premium=gross, total_premium=gross * req.exposure)
