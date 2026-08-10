from fastapi import APIRouter, HTTPException

from ...ruin import RuinTheory
from ..schemas import RuinRequest, RuinResponse

router = APIRouter(prefix="/api/ruin", tags=["ruin"])


@router.post("/analyze", response_model=RuinResponse)
def analyze_ruin(req: RuinRequest) -> RuinResponse:
    adjustment_coefficient = None
    bound = None
    exact = None

    if req.severity_dist in ('exponential', 'gamma'):
        try:
            adjustment_coefficient = RuinTheory.adjustment_coefficient(
                req.claim_rate, req.severity_dist, req.severity_params, req.premium_loading
            )
            bound = RuinTheory.ruin_probability_bound(req.initial_surplus, adjustment_coefficient)
        except ValueError:
            pass

    if req.severity_dist == 'exponential':
        try:
            exact = RuinTheory.ruin_probability_exact_exponential(
                req.initial_surplus, req.claim_rate, req.severity_params['mean'], req.premium_loading
            )
        except (ValueError, KeyError):
            pass

    model = RuinTheory(seed=req.seed)
    try:
        sim = model.simulate_finite_horizon_ruin(
            req.initial_surplus,
            req.claim_rate,
            req.severity_dist,
            req.severity_params,
            req.premium_loading,
            req.time_horizon,
            req.n_paths,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return RuinResponse(
        adjustment_coefficient=adjustment_coefficient,
        ruin_probability_bound=bound,
        ruin_probability_exact=exact,
        simulated_ruin_probability=sim['ruin_probability'],
        simulated_n_paths=sim['n_paths'],
        simulated_n_ruined=sim['n_ruined'],
        simulated_mean_time_to_ruin=sim['mean_time_to_ruin'],
    )
