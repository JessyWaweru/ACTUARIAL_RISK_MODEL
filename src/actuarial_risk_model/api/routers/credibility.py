from fastapi import APIRouter, HTTPException

from ...credibility import BuhlmannCredibility
from ..schemas import CredibilityRequest, CredibilityResponse

router = APIRouter(prefix="/api/credibility", tags=["credibility"])


@router.post("/calculate", response_model=CredibilityResponse)
def calculate_credibility(req: CredibilityRequest) -> CredibilityResponse:
    if not (0 <= req.target_index < len(req.claims_by_risk)):
        raise HTTPException(status_code=400, detail="target_index out of range")

    try:
        params = BuhlmannCredibility.estimate_parameters(req.claims_by_risk)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    target = req.claims_by_risk[req.target_index]
    individual_mean = sum(target) / len(target)
    result = BuhlmannCredibility.credibility_premium(
        individual_mean, len(target), params['epv'], params['vhm'], params['collective_mean']
    )

    return CredibilityResponse(
        epv=params['epv'],
        vhm=params['vhm'],
        collective_mean=params['collective_mean'],
        individual_mean=individual_mean,
        n=len(target),
        z=result['z'],
        credibility_premium=result['credibility_premium'],
    )
