from fastapi import APIRouter, HTTPException

from ...reserving import bornhuetter_ferguson, mack_chain_ladder
from ...risk_model import RiskModel
from ..schemas import (
    BornhuetterFergusonRequest,
    BornhuetterFergusonResponse,
    ChainLadderRequest,
    ChainLadderResponse,
    MackChainLadderResponse,
)
from ..utils import triangle_to_array

router = APIRouter(prefix="/api/reserving", tags=["reserving"])


@router.post("/chain-ladder", response_model=ChainLadderResponse)
def chain_ladder(req: ChainLadderRequest) -> ChainLadderResponse:
    triangle = triangle_to_array(req.triangle)
    model = RiskModel()
    try:
        total_reserve, dev_factors = model.chain_ladder_reserve(triangle)
    except (ValueError, ZeroDivisionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ChainLadderResponse(total_reserve=float(total_reserve), dev_factors=dev_factors.tolist())


@router.post("/chain-ladder-mack", response_model=MackChainLadderResponse)
def chain_ladder_mack(req: ChainLadderRequest) -> MackChainLadderResponse:
    triangle = triangle_to_array(req.triangle)
    try:
        result = mack_chain_ladder(triangle)
    except (ValueError, ZeroDivisionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return MackChainLadderResponse(
        dev_factors=result['dev_factors'].tolist(),
        ultimate_by_year=result['ultimate_by_year'].tolist(),
        reserve_by_year=result['reserve_by_year'].tolist(),
        standard_error_by_year=result['standard_error_by_year'].tolist(),
        total_reserve=result['total_reserve'],
        total_standard_error=result['total_standard_error'],
        coefficient_of_variation=result['coefficient_of_variation'],
    )


@router.post("/bornhuetter-ferguson", response_model=BornhuetterFergusonResponse)
def bf(req: BornhuetterFergusonRequest) -> BornhuetterFergusonResponse:
    triangle = triangle_to_array(req.triangle)
    try:
        result = bornhuetter_ferguson(triangle, req.expected_loss_ratio, req.premium)
    except (ValueError, ZeroDivisionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BornhuetterFergusonResponse(
        dev_factors=result['dev_factors'].tolist(),
        pct_reported_by_year=result['pct_reported_by_year'].tolist(),
        reserve_by_year=result['reserve_by_year'].tolist(),
        ultimate_by_year=result['ultimate_by_year'].tolist(),
        total_reserve=result['total_reserve'],
    )
