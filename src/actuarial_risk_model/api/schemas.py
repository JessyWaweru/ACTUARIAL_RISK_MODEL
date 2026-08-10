"""Pydantic request/response models for the API."""
from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

DistName = Literal['normal', 'poisson', 'lognormal', 'gamma']


# === shared ===

class SimulationInput(BaseModel):
    """Parameters to (re)generate a loss array server-side, or raw losses to use directly."""
    dist: Optional[DistName] = None
    mean: Optional[float] = None
    std_dev: Optional[float] = None
    simulations: int = 10_000
    seed: Optional[int] = None
    raw_losses: Optional[List[float]] = None


class Histogram(BaseModel):
    bin_edges: List[float]
    counts: List[int]


class LossSummary(BaseModel):
    mean: float
    std_dev: float
    min: float
    max: float
    percentiles: Dict[str, float]
    histogram: Histogram


# === premium ===

class PremiumRequest(BaseModel):
    exposure: float = Field(gt=0)
    frequency: float = Field(ge=0)
    severity: float = Field(ge=0)
    risk_load: float = 0.2
    expense_load: float = 0.15


class PremiumResponse(BaseModel):
    gross_premium: float
    total_premium: float


# === simulation ===

class MonteCarloRequest(BaseModel):
    dist: DistName
    mean: float
    std_dev: Optional[float] = None
    simulations: int = 10_000
    seed: Optional[int] = None


class MonteCarloResponse(LossSummary):
    var_95: float
    var_99: float
    tvar_95: float
    tvar_99: float


class AggregateFrequencyParams(BaseModel):
    dist: Literal['poisson', 'negative_binomial']
    mean: float
    dispersion: Optional[float] = None


class AggregateSeverityParams(BaseModel):
    dist: Literal['lognormal', 'gamma', 'pareto']
    mean: Optional[float] = None
    std_dev: Optional[float] = None
    alpha: Optional[float] = None
    scale: Optional[float] = None


class AggregateLossRequest(BaseModel):
    frequency: AggregateFrequencyParams
    severity: AggregateSeverityParams
    simulations: int = 10_000
    seed: Optional[int] = None


class AggregateLossResponse(LossSummary):
    mean_claim_count: float
    var_95: float
    var_99: float
    tvar_95: float
    tvar_99: float


# === risk metrics ===

class RiskMetricsRequest(BaseModel):
    simulation: SimulationInput
    confidence: float = 0.95


class RiskMetricsResponse(BaseModel):
    mean: float
    std_dev: float
    var: float
    tvar: float
    max_loss: float
    min_loss: float
    confidence: float


# === reinsurance ===

class ReinsuranceLayerRequest(BaseModel):
    simulation: SimulationInput
    attachment: float = Field(ge=0)
    limit: float = Field(gt=0)


class ReinsuranceLayerResponse(BaseModel):
    pure_premium: float
    risk_load: float
    gross_premium: float
    loss_ratio: float


class RateOnLineRequest(BaseModel):
    premium: float = Field(gt=0)
    limit: float = Field(gt=0)


class RateOnLineResponse(BaseModel):
    rate_on_line: float


class XolReinstatementRequest(BaseModel):
    annual_occurrence_losses: List[List[float]]
    attachment: float = Field(ge=0)
    limit: float = Field(gt=0)
    num_reinstatements: int = Field(ge=0)
    reinstatement_cost_pct: float = 1.0
    risk_load_factor: float = 0.2


class XolReinstatementResponse(BaseModel):
    first_layer_pure_premium: float
    risk_load: float
    first_layer_premium: float
    rate_on_line: float
    expected_reinstatements_used: float
    reinstatement_premium: float
    gross_premium: float
    effective_rate_on_line: float
    expected_total_recovery: float
    loss_ratio: float


# === reserving ===

class ChainLadderRequest(BaseModel):
    triangle: List[List[Optional[float]]]


class ChainLadderResponse(BaseModel):
    total_reserve: float
    dev_factors: List[float]


class MackChainLadderResponse(BaseModel):
    dev_factors: List[float]
    ultimate_by_year: List[float]
    reserve_by_year: List[float]
    standard_error_by_year: List[float]
    total_reserve: float
    total_standard_error: float
    coefficient_of_variation: float


class BornhuetterFergusonRequest(BaseModel):
    triangle: List[List[Optional[float]]]
    expected_loss_ratio: float = Field(gt=0)
    premium: List[float]


class BornhuetterFergusonResponse(BaseModel):
    dev_factors: List[float]
    pct_reported_by_year: List[float]
    reserve_by_year: List[float]
    ultimate_by_year: List[float]
    total_reserve: float


# === credibility ===

class CredibilityRequest(BaseModel):
    claims_by_risk: List[List[float]]
    target_index: int = 0


class CredibilityResponse(BaseModel):
    epv: float
    vhm: float
    collective_mean: float
    individual_mean: float
    n: int
    z: float
    credibility_premium: float


# === portfolio ===

class PortfolioLine(BaseModel):
    name: str
    dist: Literal['lognormal', 'gamma']
    mean: float = Field(gt=0)
    std_dev: float = Field(gt=0)


class PortfolioRequest(BaseModel):
    lines: List[PortfolioLine]
    correlation_matrix: List[List[float]]
    simulations: int = 10_000
    seed: Optional[int] = None


class PortfolioResponse(BaseModel):
    line_names: List[str]
    per_line_mean: List[float]
    per_line_std: List[float]
    correlated_std: float
    sum_of_individual_std: float
    diversification_benefit_pct: float
    total_histogram: Histogram


# === extreme value ===

class ExtremeValueRequest(BaseModel):
    simulation: SimulationInput
    threshold: float
    return_periods: List[float] = [10, 50, 100, 250]
    confidence: float = 0.99
    events_per_period: float = 1.0


class ReturnLevelPoint(BaseModel):
    return_period: float
    level: float


class ExtremeValueResponse(BaseModel):
    shape: float
    scale: float
    threshold: float
    n_exceedances: int
    exceedance_rate: float
    return_levels: List[ReturnLevelPoint]
    tail_var: float
    tail_tvar: float


# === ruin ===

class RuinRequest(BaseModel):
    initial_surplus: float = Field(ge=0)
    claim_rate: float = Field(gt=0)
    severity_dist: Literal['exponential', 'gamma', 'lognormal']
    severity_params: Dict[str, float]
    premium_loading: float = Field(gt=0)
    time_horizon: float = Field(gt=0)
    n_paths: int = 5_000
    seed: Optional[int] = None


class RuinResponse(BaseModel):
    adjustment_coefficient: Optional[float] = None
    ruin_probability_bound: Optional[float] = None
    ruin_probability_exact: Optional[float] = None
    simulated_ruin_probability: float
    simulated_n_paths: int
    simulated_n_ruined: int
    simulated_mean_time_to_ruin: Optional[float] = None


# === sensitivity ===

class PremiumSensitivityRequest(BaseModel):
    base: PremiumRequest
    param_name: Literal['exposure', 'frequency', 'severity', 'risk_load', 'expense_load']
    values: List[float]


class VarSensitivityRequest(BaseModel):
    base: MonteCarloRequest
    param_name: Literal['mean', 'std_dev']
    values: List[float]
    confidence: float = 0.95


class SensitivityPoint(BaseModel):
    value: float
    output: float


class SensitivityResponse(BaseModel):
    parameter: str
    base_value: Optional[float] = None
    base_output: float
    results: List[SensitivityPoint]


# === runs (saved history) ===

class RunCreateRequest(BaseModel):
    kind: str
    name: str
    input: dict
    result: dict


class RunSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    kind: str
    name: str
    created_at: datetime


class RunDetail(RunSummary):
    input: dict
    result: dict
