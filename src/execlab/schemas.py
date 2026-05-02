from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator


Side = Literal["buy", "sell"]
Interval = Literal["1m", "5m", "15m"]
AlgoName = Literal["TWAP", "VWAP", "POV", "IS"]
PovMode = Literal["strict_cap", "force_complete"]


class ExecutionRequest(BaseModel):
    ticker: str = Field(default="NVDA", min_length=1, max_length=12)
    trade_date: date
    side: Side = "buy"
    quantity: int = Field(default=50_000, gt=0)
    start_time: time = Field(default=time(9, 30))
    end_time: time = Field(default=time(16, 0))
    interval: Interval = "5m"
    algos: list[AlgoName] = Field(default_factory=lambda: ["TWAP", "VWAP", "POV", "IS"])
    participation_rate: float = Field(default=0.10, gt=0, le=1.0)
    pov_mode: PovMode = "strict_cap"
    urgency: float = Field(default=0.65, ge=0.0, le=1.0)
    limit_price: float | None = Field(default=None, gt=0)
    spread_bps: float = Field(default=2.0, ge=0.0, le=100.0)
    impact_bps_per_10pct: float = Field(default=1.5, ge=0.0, le=100.0)
    drift_bps_per_day: float = Field(default=0.0, ge=-500.0, le=500.0)
    scenario_paths: int = Field(default=300, ge=50, le=2000)
    seed: int = Field(default=4576)
    custom_algo_instructions: str = Field(default="", max_length=4000)
    user_id: str = "local-user"

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("algos")
    @classmethod
    def require_algos(cls, value: list[AlgoName]) -> list[AlgoName]:
        deduped: list[AlgoName] = []
        for algo in value:
            if algo not in deduped:
                deduped.append(algo)
        if not deduped:
            raise ValueError("At least one algorithm must be selected.")
        return deduped

    @field_validator("custom_algo_instructions")
    @classmethod
    def clean_custom_algo_instructions(cls, value: str) -> str:
        return " ".join(str(value or "").split())

    @field_validator("end_time")
    @classmethod
    def require_forward_window(cls, value: time, info) -> time:
        start = info.data.get("start_time")
        if start and value <= start:
            raise ValueError("end_time must be after start_time.")
        return value


class BarRecord(BaseModel):
    timestamp_et: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataPayload(BaseModel):
    ticker: str
    trade_date: date
    interval: str
    provider: str
    row_count: int
    records: list[BarRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class MarketEDA(BaseModel):
    ticker: str
    trade_date: date
    interval: str
    bar_count: int
    arrival_price: float
    close_price: float
    market_vwap: float
    total_volume: float
    window_volume: float
    price_move_bps: float
    realized_volatility_bps: float
    high_low_spread_proxy_bps: float
    volume_curve_source: str


class ScheduleSummary(BaseModel):
    algo: str
    parent_quantity: int = 0
    total_quantity: int
    child_order_count: int
    max_child_order: int
    max_participation_rate: float
    unfilled_quantity: int = 0
    completion_rate: float = 1.0
    cap_violation_count: int = 0
    schedule_note: str = ""


class FillRecord(BaseModel):
    timestamp_et: datetime
    algo: str
    target_quantity: int
    executed_quantity: int
    unfilled_quantity: int = 0
    bar_volume: float
    market_price: float
    fill_price: float
    participation_rate: float
    executable: bool = True
    blocked_reason: str | None = None


class TcaMetrics(BaseModel):
    algo: str
    avg_fill_price: float
    arrival_price: float
    market_vwap: float
    close_price: float
    arrival_cost_bps: float
    vwap_slippage_bps: float
    close_slippage_bps: float
    total_quantity_targeted: int = 0
    total_quantity_executed: int
    unfilled_quantity: int = 0
    completion_rate: float = 1.0
    max_participation_rate: float
    cap_violation_count: int = 0


class SimulationResult(BaseModel):
    algo: str
    fills: list[FillRecord]
    metrics: TcaMetrics
    schedule_summary: ScheduleSummary


class ScenarioAlgoResult(BaseModel):
    algo: str
    expected_arrival_cost_bps: float
    p10_arrival_cost_bps: float
    p50_arrival_cost_bps: float
    p90_arrival_cost_bps: float
    probability_cost_positive: float


class ScenarioReport(BaseModel):
    path_count: int
    seed: int
    spread_bps: float
    impact_bps_per_10pct: float
    drift_bps_per_day: float
    results: list[ScenarioAlgoResult] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class PreTradeCurvePoint(BaseModel):
    time_key: str
    avg_volume: float
    avg_volume_pct: float
    avg_spread_proxy_bps: float
    avg_volatility_bps: float
    time_risk_score: float


class RegressionCoefficient(BaseModel):
    feature: str
    coefficient: float
    units: str = "bps contribution per feature unit"


class CostBreakdownItem(BaseModel):
    component: str
    bps: float
    description: str


class ThroughDayCostPoint(BaseModel):
    time_key: str
    p10_bps: float
    p50_bps: float
    p90_bps: float


class PreTradeAnalyticsReport(BaseModel):
    lookback_sessions: int
    requested_sessions: int
    adv_shares: float
    order_size_adv_pct: float
    avg_spread_proxy_bps: float
    avg_volatility_bps: float
    time_risk_score: float
    current_vs_21d_volume: float
    curve: list[PreTradeCurvePoint] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ExpectedCostModelReport(BaseModel):
    expected_cost_bps: float
    model_r2: float
    observation_count: int
    coefficients: list[RegressionCoefficient] = Field(default_factory=list)
    cost_breakdown: list[CostBreakdownItem] = Field(default_factory=list)
    through_day_cost: list[ThroughDayCostPoint] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class BetaRiskReport(BaseModel):
    ticker: str
    market_etf: str = "SPY"
    sector_etf: str = "SPY"
    sector_label: str = "Broad market"
    mapping_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    mapping_reason: str = ""
    lookback_days: int
    observation_count: int
    beta_market: float
    beta_sector: float
    correlation_market: float
    correlation_sector: float
    r_squared: float
    ticker_daily_vol_bps: float
    market_daily_vol_bps: float
    sector_daily_vol_bps: float
    systematic_risk_bps: float
    sector_risk_bps: float
    idiosyncratic_risk_bps: float
    total_timing_risk_bps: float
    systematic_cost_bps: float
    idiosyncratic_cost_bps: float
    coefficients: list[RegressionCoefficient] = Field(default_factory=list)
    index_comparison: list["IndexComparisonPoint"] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class IndexComparisonPoint(BaseModel):
    time_key: str
    ticker_return_bps: float
    market_return_bps: float
    sector_return_bps: float


class PeerStockMetric(BaseModel):
    ticker: str
    correlation: float
    beta_to_target: float
    recent_move_bps: float
    target_move_bps: float
    move_gap_bps: float
    cluster: str
    impact_signal: str


class PeerStockAnalysisReport(BaseModel):
    sector_etf: str
    candidate_count: int
    analyzed_count: int
    average_peer_correlation: float
    crowding_score: float
    median_peer_move_bps: float
    target_recent_move_bps: float
    urgency_recommendation: str
    rationale: str
    market_impact_note: str
    peers: list[PeerStockMetric] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TcaCausalBullet(BaseModel):
    driver: str
    affected_algos: list[str] = Field(default_factory=list)
    evidence: str
    implication: str


class TcaCausalReport(BaseModel):
    best_algo: str
    headline: str
    bullets: list[TcaCausalBullet] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class DebateCase(BaseModel):
    advocate: str
    stance: str
    recommended_algos: list[str] = Field(default_factory=list)
    thesis: str
    evidence: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class DebateReport(BaseModel):
    fast_case: DebateCase
    liquidity_case: DebateCase
    judge_winner: str
    recommended_algo: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    deciding_factors: list[str] = Field(default_factory=list)
    judge_rationale: str


class CounterfactualScenario(BaseModel):
    name: str
    assumption_change: str
    estimated_winner: str
    current_winner: str
    estimated_costs_bps: dict[str, float] = Field(default_factory=dict)
    rationale: str


class CounterfactualReport(BaseModel):
    base_winner: str
    scenarios: list[CounterfactualScenario] = Field(default_factory=list)
    summary: str


class ExecutionPlaybookReport(BaseModel):
    recommended_algo: str
    urgency: str
    participation_guidance: str
    limit_guidance: str
    monitoring_triggers: list[str] = Field(default_factory=list)
    switch_rules: list[str] = Field(default_factory=list)
    rationale: str


class CustomAlgoComponent(BaseModel):
    name: str
    weight: float = Field(ge=0.0, le=1.0)
    reason: str


class CustomAlgoPlan(BaseModel):
    status: str = Field(description="ok, needs_clarification, or warning")
    objective_summary: str = Field(
        description="Concise summary of the user's execution objective."
    )
    urgency_score: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Agent-read urgency from the user's brief and market context.",
    )
    liquidity_score: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Agent-read need to minimize footprint and seek liquidity.",
    )
    max_participation_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Maximum participation cap requested by the user, if any.",
    )
    completion_target_pct: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Share of the order the user wants completed by a target time.",
    )
    completion_target_time: str | None = Field(
        default=None,
        description="Target completion clock time such as 11:00 or 14:30.",
    )
    must_complete: bool = False
    strict_cap: bool = False
    pm_exposure_summary: str = ""
    risk_constraints: list[str] = Field(default_factory=list)
    style_hint: str = Field(
        default="adaptive_hybrid",
        description="Suggested style such as liquidity_seeker, front_loaded_is, adaptive_vwap, or limit_aware.",
    )
    component_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Optional weights for vwap_curve, is_urgency, pov_guardrail, and twap_stabilizer.",
    )
    rationale: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)


class CustomAlgoReport(BaseModel):
    name: str
    style: str
    description: str
    components: list[CustomAlgoComponent] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    rationale: list[str] = Field(default_factory=list)
    simulation: SimulationResult
    caveats: list[str] = Field(default_factory=list)


class AgentStepReport(BaseModel):
    status: str = Field(
        description="Short status such as ok, warning, or review."
    )
    highlights: list[str] = Field(
        default_factory=list,
        description=(
            "Three to five visible analyst reasoning bullets. Each bullet should connect a "
            "computed observation to the execution implication. Do not include hidden chain-of-thought."
        ),
    )
    caveats: list[str] = Field(
        default_factory=list,
        description="Important limitations, data caveats, or risk warnings for this agent section.",
    )


class ExecutionMemo(BaseModel):
    best_algo: str
    thesis: str
    evidence: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    scenario_interpretation: str = ""
    limitation: str = (
        "This is a bar-based execution research simulator, not a production OMS/EMS backtester."
    )


@dataclass
class RunResult:
    request: ExecutionRequest
    provider: str
    bars: pd.DataFrame
    window_bars: pd.DataFrame
    schedules: dict[str, pd.DataFrame]
    simulations: dict[str, SimulationResult]
    eda: MarketEDA
    pretrade_report: PreTradeAnalyticsReport
    expected_cost_report: ExpectedCostModelReport
    beta_risk_report: BetaRiskReport
    peer_report: PeerStockAnalysisReport
    causal_report: TcaCausalReport
    debate_report: DebateReport
    counterfactual_report: CounterfactualReport
    playbook_report: ExecutionPlaybookReport
    custom_schedule: pd.DataFrame
    custom_algo_report: CustomAlgoReport
    scenario_report: ScenarioReport
    memo: ExecutionMemo
    agent_reports: dict[str, AgentStepReport] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    adk_status: str = "not_attempted"
    adk_attempted: bool = False
    adk_error_summary: str | None = None
    adk_model_used: str | None = None
    execution_trace: list[dict[str, Any]] = field(default_factory=list)
    runtime_seconds: float = 0.0
