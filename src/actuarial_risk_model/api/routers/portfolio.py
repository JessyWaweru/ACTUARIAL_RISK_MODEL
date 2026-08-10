import numpy as np
from fastapi import APIRouter, HTTPException

from ...portfolio import CorrelatedPortfolio
from ..schemas import PortfolioRequest, PortfolioResponse
from ..utils import build_histogram

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])


@router.post("/simulate", response_model=PortfolioResponse)
def simulate_portfolio(req: PortfolioRequest) -> PortfolioResponse:
    model = CorrelatedPortfolio(seed=req.seed)
    lines = [line.model_dump() for line in req.lines]
    try:
        result = model.simulate(lines, np.array(req.correlation_matrix), n_simulations=req.simulations)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    per_line = result['per_line_losses']
    return PortfolioResponse(
        line_names=result['line_names'],
        per_line_mean=np.mean(per_line, axis=0).tolist(),
        per_line_std=np.std(per_line, axis=0).tolist(),
        correlated_std=result['correlated_std'],
        sum_of_individual_std=result['sum_of_individual_std'],
        diversification_benefit_pct=result['diversification_benefit_pct'],
        total_histogram=build_histogram(result['total_losses']),
    )
