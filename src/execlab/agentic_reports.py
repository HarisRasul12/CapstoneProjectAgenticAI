from __future__ import annotations

import numpy as np
import pandas as pd

from execlab.schemas import (
    BetaRiskReport,
    CounterfactualReport,
    CounterfactualScenario,
    DebateCase,
    DebateReport,
    ExecutionPlaybookReport,
    ExecutionRequest,
    PeerStockAnalysisReport,
    PreTradeAnalyticsReport,
    SimulationResult,
)


def build_debate_report(
    request: ExecutionRequest,
    simulations: dict[str, SimulationResult],
    schedules: dict[str, pd.DataFrame],
    pretrade: PreTradeAnalyticsReport,
    beta_risk: BetaRiskReport,
    peer_report: PeerStockAnalysisReport,
) -> DebateReport:
    ranked = _rank_simulations(simulations)
    best = ranked[0]
    worst = ranked[-1]
    fast_algos = [algo for algo in ["IS", "POV"] if algo in simulations]
    liquidity_algos = [algo for algo in ["VWAP", "TWAP", "POV"] if algo in simulations]
    best_fast = _best_algo(simulations, fast_algos) or best.algo
    best_liquidity = _best_algo(simulations, liquidity_algos) or best.algo
    adverse_move = _is_adverse_move(request, _price_move_from_metrics(best))
    systematic_heavy = beta_risk.systematic_risk_bps + beta_risk.sector_risk_bps > beta_risk.idiosyncratic_risk_bps
    high_friction = pretrade.avg_spread_proxy_bps >= 5.0 or pretrade.order_size_adv_pct >= 0.08

    fast_case = DebateCase(
        advocate="FastExecutionAdvocate",
        stance="Trade earlier to reduce market timing risk.",
        recommended_algos=[best_fast],
        thesis=(
            f"{best_fast} is the strongest fast-execution candidate because adverse timing, beta risk, "
            "or peer crowding can make waiting expensive."
        ),
        evidence=[
            f"Best fast candidate {best_fast}: {simulations[best_fast].metrics.arrival_cost_bps:.2f} bps versus arrival.",
            f"Window timing risk: systematic {beta_risk.systematic_risk_bps:.2f} bps, sector {beta_risk.sector_risk_bps:.2f} bps, idiosyncratic {beta_risk.idiosyncratic_risk_bps:.2f} bps.",
            f"Peer urgency read: {peer_report.urgency_recommendation} with crowding score {peer_report.crowding_score:.2f}.",
        ],
        caveats=[
            "Faster schedules can pay more spread and impact when participation is high.",
            "A favorable tape can make slower schedules look better after the fact.",
        ],
    )

    liquidity_case = DebateCase(
        advocate="LiquiditySeekingAdvocate",
        stance="Stay close to natural liquidity and avoid unnecessary footprint.",
        recommended_algos=[best_liquidity],
        thesis=(
            f"{best_liquidity} is the strongest liquidity-seeking candidate because it balances "
            "benchmark tracking, participation, and spread/impact control."
        ),
        evidence=[
            f"Best liquidity candidate {best_liquidity}: {simulations[best_liquidity].metrics.arrival_cost_bps:.2f} bps versus arrival.",
            f"21-day spread proxy {pretrade.avg_spread_proxy_bps:.2f} bps and order size {pretrade.order_size_adv_pct * 100:.2f}% of ADV.",
            f"Current volume versus 21-day curve: {pretrade.current_vs_21d_volume:.2f}x.",
        ],
        caveats=[
            "Slower schedules retain timing exposure if the market or peers move adversely.",
            "VWAP quality depends on the volume-curve source and live volume regime.",
        ],
    )

    gap = abs(best.metrics.arrival_cost_bps - worst.metrics.arrival_cost_bps)
    if best.algo in fast_algos and (adverse_move or systematic_heavy or peer_report.urgency_recommendation.startswith("Faster")):
        judge_winner = "FastExecutionAdvocate"
    elif high_friction or best.algo in liquidity_algos:
        judge_winner = "LiquiditySeekingAdvocate"
    else:
        judge_winner = "FastExecutionAdvocate" if best.algo in fast_algos else "LiquiditySeekingAdvocate"

    confidence = float(min(0.95, 0.45 + gap / 25.0))
    deciding_factors = [
        f"Current arrival-cost leader: {best.algo} at {best.metrics.arrival_cost_bps:.2f} bps.",
        f"Spread/impact friction: {pretrade.avg_spread_proxy_bps:.2f} bps spread proxy and {pretrade.order_size_adv_pct * 100:.2f}% ADV order size.",
        f"Peer crowding: {peer_report.crowding_score:.2f}; peer recommendation: {peer_report.urgency_recommendation}.",
    ]
    rationale = (
        f"The judge selects {best.algo} because it has the best completed arrival-cost result while the "
        f"risk/peer context favors {judge_winner.replace('Advocate', '').lower()} discipline."
    )
    return DebateReport(
        fast_case=fast_case,
        liquidity_case=liquidity_case,
        judge_winner=judge_winner,
        recommended_algo=best.algo,
        confidence=confidence,
        deciding_factors=deciding_factors,
        judge_rationale=rationale,
    )


def build_counterfactual_report(
    request: ExecutionRequest,
    simulations: dict[str, SimulationResult],
    schedules: dict[str, pd.DataFrame],
    pretrade: PreTradeAnalyticsReport,
    peer_report: PeerStockAnalysisReport,
) -> CounterfactualReport:
    base_costs = {algo: sim.metrics.arrival_cost_bps for algo, sim in simulations.items()}
    base_winner = min(base_costs, key=base_costs.get)
    scenarios: list[CounterfactualScenario] = []

    scenarios.append(
        _scenario_from_costs(
            name="Flat tape",
            assumption_change="Remove the realized price-path timing component from each schedule.",
            current_winner=base_winner,
            costs={
                algo: cost - _timing_component(request, schedules.get(algo), simulations[algo])
                for algo, cost in base_costs.items()
            },
            rationale="Shows whether the winner depended on realized drift rather than structural liquidity quality.",
        )
    )

    scenarios.append(
        _scenario_from_costs(
            name="Spread widens by 5 bps",
            assumption_change="Add a 5 bps spread shock, with extra penalty for high participation.",
            current_winner=base_winner,
            costs={
                algo: cost + 2.5 + 5.0 * simulations[algo].metrics.max_participation_rate
                for algo, cost in base_costs.items()
            },
            rationale="Tests whether the result survives a more expensive liquidity environment.",
        )
    )

    scenarios.append(
        _scenario_from_costs(
            name="Order size doubles",
            assumption_change="Add impact pressure proportional to each algo's max participation.",
            current_winner=base_winner,
            costs={
                algo: cost
                + request.impact_bps_per_10pct * max(1.0, simulations[algo].metrics.max_participation_rate / 0.10)
                for algo, cost in base_costs.items()
            },
            rationale="Larger orders punish schedules that concentrate volume in fewer bars.",
        )
    )

    adverse_peer_penalty = abs(peer_report.median_peer_move_bps) * max(0.0, peer_report.crowding_score) * 0.08
    scenarios.append(
        _scenario_from_costs(
            name="Adverse peer crowding",
            assumption_change="Apply a timing-risk penalty to slower schedules when correlated peers move together.",
            current_winner=base_winner,
            costs={
                algo: cost + adverse_peer_penalty * _weighted_time_fraction(schedules.get(algo))
                for algo, cost in base_costs.items()
            },
            rationale="Shows whether peer crowding would argue for faster execution.",
        )
    )

    scenarios.append(
        _scenario_from_costs(
            name="Completion-adjusted view",
            assumption_change="Penalize unfilled shares at 25 bps per unfilled percentage point.",
            current_winner=base_winner,
            costs={
                algo: cost + 25.0 * (1.0 - simulations[algo].metrics.completion_rate)
                for algo, cost in base_costs.items()
            },
            rationale="Prevents a low-cost but incomplete schedule from looking falsely superior.",
        )
    )

    flip_count = sum(1 for scenario in scenarios if scenario.estimated_winner != base_winner)
    summary = (
        f"{flip_count} of {len(scenarios)} counterfactuals change the winner from {base_winner}. "
        "A stable winner is more robust; frequent flips mean the recommendation depends on assumptions."
    )
    return CounterfactualReport(base_winner=base_winner, scenarios=scenarios, summary=summary)


def build_execution_playbook(
    request: ExecutionRequest,
    debate: DebateReport,
    counterfactuals: CounterfactualReport,
    pretrade: PreTradeAnalyticsReport,
    beta_risk: BetaRiskReport,
    peer_report: PeerStockAnalysisReport,
) -> ExecutionPlaybookReport:
    fast = debate.judge_winner == "FastExecutionAdvocate"
    flip_count = sum(
        1 for scenario in counterfactuals.scenarios if scenario.estimated_winner != counterfactuals.base_winner
    )
    if fast:
        urgency = "Use a front-loaded or higher-urgency profile; review after the first 20-30% of the order."
    elif debate.recommended_algo == "POV":
        urgency = "Use strict capped POV unless completion becomes more important than footprint."
    else:
        urgency = "Stay near VWAP/TWAP pacing and let natural liquidity absorb the order."

    participation = (
        f"Start near {request.participation_rate * 100:.0f}% POV. "
        f"Do not exceed {max(request.participation_rate * 100, 15):.0f}% without an explicit force-complete decision."
    )
    if pretrade.order_size_adv_pct > 0.10:
        participation += " The order is large versus ADV, so participation should be staged and monitored."

    if request.limit_price is None:
        limit = "No limit is active; add a soft guardrail if price moves beyond the arrival-cost tolerance."
    else:
        limit = f"Respect limit {request.limit_price:.4f}; if completion falls behind, decide whether to loosen the limit or carry residual shares."

    monitoring = [
        f"Pause or slow down if spread proxy widens above {max(5.0, pretrade.avg_spread_proxy_bps * 1.5):.2f} bps.",
        f"Increase urgency if {request.ticker} and peers keep moving adversely and crowding remains above 0.55.",
        "Switch away from strict POV if completion rate falls below 80% near the final third of the window.",
        f"Re-check beta risk if {beta_risk.market_etf}/{beta_risk.sector_etf} moves accelerate versus the stock.",
    ]
    switch_rules = [
        "VWAP -> IS if adverse price move accelerates and peer crowding confirms.",
        "IS -> VWAP if spread/impact dominates and the tape stops moving adversely.",
        "POV strict -> force-complete only when completion matters more than cap discipline.",
    ]
    rationale = (
        f"Playbook chooses {debate.recommended_algo} with {debate.confidence:.0%} debate confidence. "
        f"{flip_count} counterfactuals changed the winner, so robustness should be monitored live."
    )
    return ExecutionPlaybookReport(
        recommended_algo=debate.recommended_algo,
        urgency=urgency,
        participation_guidance=participation,
        limit_guidance=limit,
        monitoring_triggers=monitoring,
        switch_rules=switch_rules,
        rationale=rationale,
    )


def _rank_simulations(simulations: dict[str, SimulationResult]) -> list[SimulationResult]:
    return sorted(simulations.values(), key=lambda sim: sim.metrics.arrival_cost_bps)


def _best_algo(simulations: dict[str, SimulationResult], algos: list[str]) -> str | None:
    available = [algo for algo in algos if algo in simulations]
    if not available:
        return None
    return min(available, key=lambda algo: simulations[algo].metrics.arrival_cost_bps)


def _price_move_from_metrics(simulation: SimulationResult) -> float:
    arrival = simulation.metrics.arrival_price
    close = simulation.metrics.close_price
    if arrival <= 0:
        return 0.0
    return (close / arrival - 1.0) * 10_000


def _is_adverse_move(request: ExecutionRequest, price_move_bps: float) -> bool:
    return (request.side == "buy" and price_move_bps > 0) or (
        request.side == "sell" and price_move_bps < 0
    )


def _weighted_time_fraction(schedule: pd.DataFrame | None) -> float:
    if schedule is None or schedule.empty or float(schedule["target_quantity"].sum()) <= 0:
        return 0.0
    frame = schedule.reset_index(drop=True)
    if len(frame) <= 1:
        return 0.0
    positions = pd.Series(range(len(frame)), dtype=float) / (len(frame) - 1)
    return float((positions * frame["target_quantity"].astype(float)).sum() / frame["target_quantity"].sum())


def _timing_component(
    request: ExecutionRequest,
    schedule: pd.DataFrame | None,
    simulation: SimulationResult,
) -> float:
    price_move = _price_move_from_metrics(simulation)
    side_sign = 1.0 if request.side == "buy" else -1.0
    return float(side_sign * price_move * _weighted_time_fraction(schedule))


def _scenario_from_costs(
    name: str,
    assumption_change: str,
    current_winner: str,
    costs: dict[str, float],
    rationale: str,
) -> CounterfactualScenario:
    rounded = {algo: round(float(value), 2) for algo, value in sorted(costs.items())}
    winner = min(rounded, key=rounded.get) if rounded else "Unavailable"
    return CounterfactualScenario(
        name=name,
        assumption_change=assumption_change,
        estimated_winner=winner,
        current_winner=current_winner,
        estimated_costs_bps=rounded,
        rationale=rationale,
    )
