import numpy as np
from fastapi import APIRouter, HTTPException

from ...aggregate_loss import AggregateLossModel
from ...risk_model import RiskModel
from ..schemas import AggregateLossRequest, AggregateLossResponse, MonteCarloRequest, MonteCarloResponse
from ..utils import build_loss_summary

router = APIRouter(prefix="/api/simulation", tags=["simulation"])


@router.post("/monte-carlo", response_model=MonteCarloResponse)
def run_monte_carlo(req: MonteCarloRequest) -> MonteCarloResponse:
    model = RiskModel(seed=req.seed)
    params = {'mean': req.mean}
    if req.dist in ('normal', 'lognormal', 'gamma'):
        if req.std_dev is None:
            raise HTTPException(status_code=400, detail=f"std_dev is required for {req.dist} distribution")
        params['std_dev'] = req.std_dev

    try:
        losses = model.monte_carlo_simulation(req.dist, params, simulations=req.simulations)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    summary = build_loss_summary(losses)
    return MonteCarloResponse(
        **summary,
        var_95=float(model.calculate_var(losses, 0.95)),
        var_99=float(model.calculate_var(losses, 0.99)),
        tvar_95=float(model.calculate_tvar(losses, 0.95)),
        tvar_99=float(model.calculate_tvar(losses, 0.99)),
    )


@router.post("/aggregate-loss", response_model=AggregateLossResponse)
def run_aggregate_loss(req: AggregateLossRequest) -> AggregateLossResponse:
    model = AggregateLossModel(seed=req.seed)

    freq_params = {'mean': req.frequency.mean}
    if req.frequency.dispersion is not None:
        freq_params['dispersion'] = req.frequency.dispersion

    sev_params = {}
    if req.severity.dist in ('lognormal', 'gamma'):
        if req.severity.mean is None or req.severity.std_dev is None:
            raise HTTPException(status_code=400, detail="mean and std_dev are required for this severity distribution")
        sev_params = {'mean': req.severity.mean, 'std_dev': req.severity.std_dev}
    elif req.severity.dist == 'pareto':
        if req.severity.alpha is None or req.severity.scale is None:
            raise HTTPException(status_code=400, detail="alpha and scale are required for pareto severity")
        sev_params = {'alpha': req.severity.alpha, 'scale': req.severity.scale}

    try:
        result = model.simulate(
            req.frequency.dist, freq_params, req.severity.dist, sev_params, n_simulations=req.simulations
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    losses = result['aggregate_losses']
    risk_model = RiskModel()
    summary = build_loss_summary(losses)
    return AggregateLossResponse(
        **summary,
        mean_claim_count=float(np.mean(result['claim_counts'])),
        var_95=float(risk_model.calculate_var(losses, 0.95)),
        var_99=float(risk_model.calculate_var(losses, 0.99)),
        tvar_95=float(risk_model.calculate_tvar(losses, 0.95)),
        tvar_99=float(risk_model.calculate_tvar(losses, 0.99)),
    )
