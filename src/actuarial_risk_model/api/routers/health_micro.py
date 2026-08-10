from fastapi import APIRouter, HTTPException

from ...health_micro import HealthMicroReserving, build_illustrative_triangle
from ..schemas import HealthCatCoverRequest, HealthCatCoverResponse, HealthTriangleRequest, HealthTriangleResponse
from ..utils import build_histogram

router = APIRouter(prefix="/api/health-micro", tags=["health-micro"])


@router.post("/triangle", response_model=HealthTriangleResponse)
def triangle(req: HealthTriangleRequest) -> HealthTriangleResponse:
    tri = build_illustrative_triangle(req.n_years, req.base_claims, req.growth, seed=req.seed)
    result = HealthMicroReserving.reserve(tri)

    return HealthTriangleResponse(
        triangle=[[None if v != v else float(v) for v in row] for row in tri],  # v != v <=> NaN
        dev_factors=result['dev_factors'].tolist(),
        ultimate_by_year=result['ultimate_by_year'].tolist(),
        reserve_by_year=result['reserve_by_year'].tolist(),
        standard_error_by_year=result['standard_error_by_year'].tolist(),
        total_reserve=result['total_reserve'],
        total_standard_error=result['total_standard_error'],
        coefficient_of_variation=result['coefficient_of_variation'],
    )


@router.post("/catastrophic-cover", response_model=HealthCatCoverResponse)
def catastrophic_cover(req: HealthCatCoverRequest) -> HealthCatCoverResponse:
    if req.limit <= 0:
        raise HTTPException(status_code=400, detail="limit must be positive")
    claims = HealthMicroReserving.simulate_annual_claims(req.mean_annual_claims, req.cv, req.n_years, req.seed)
    pricing = HealthMicroReserving.catastrophic_cover_pricing(claims, req.deductible, req.limit)

    return HealthCatCoverResponse(
        pure_premium=pricing['pure_premium'],
        risk_load=pricing['risk_load'],
        gross_premium=pricing['gross_premium'],
        loss_ratio=pricing['loss_ratio'],
        histogram=build_histogram(claims),
    )
