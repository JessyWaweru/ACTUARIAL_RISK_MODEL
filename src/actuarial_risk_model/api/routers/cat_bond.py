import numpy as np
from fastapi import APIRouter, HTTPException

from ...cat_bond import FloodCatBond, flood_payout_fraction, load_daily_rainfall
from ...extreme_value import ExtremeValueModel
from ..schemas import CatBondRequest, CatBondResponse
from ..utils import build_histogram

router = APIRouter(prefix="/api/cat-bond", tags=["cat-bond"])


@router.post("/price", response_model=CatBondResponse)
def price(req: CatBondRequest) -> CatBondResponse:
    rainfall = np.array(load_daily_rainfall())
    threshold = float(np.percentile(rainfall, req.threshold_percentile))

    try:
        fit = ExtremeValueModel.fit_gpd(rainfall, threshold)
        annual_max = FloodCatBond.simulate_annual_max_rainfall(
            fit['shape'], fit['scale'], fit['threshold'], fit['exceedance_rate'], req.n_years, req.seed
        )
        pricing = FloodCatBond.price_bond(
            annual_max, req.attachment_mm, req.exhaustion_mm, req.principal, req.risk_multiple
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payout_amounts = flood_payout_fraction(annual_max, req.attachment_mm, req.exhaustion_mm) * req.principal

    return CatBondResponse(
        threshold_mm=fit['threshold'],
        shape=fit['shape'],
        scale=fit['scale'],
        exceedance_rate=fit['exceedance_rate'],
        n_exceedances=fit['n_exceedances'],
        expected_loss_pct=pricing['expected_loss_pct'],
        expected_loss_amount=pricing['expected_loss_amount'],
        probability_of_attachment=pricing['probability_of_attachment'],
        probability_of_exhaustion=pricing['probability_of_exhaustion'],
        coupon_spread_pct=pricing['coupon_spread_pct'],
        annual_coupon_amount=pricing['annual_coupon_amount'],
        histogram=build_histogram(payout_amounts),
    )
