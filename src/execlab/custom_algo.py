from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

from execlab.schemas import (
    BetaRiskReport,
    CustomAlgoComponent,
    CustomAlgoPlan,
    CustomAlgoReport,
    DebateReport,
    ExecutionRequest,
    MarketEDA,
    PeerStockAnalysisReport,
    PreTradeAnalyticsReport,
)
from execlab.simulator import simulate_fills


def build_custom_algo_report(
    request: ExecutionRequest,
    bars: pd.DataFrame,
    volume_curve: pd.Series,
    eda: MarketEDA,
    pretrade: PreTradeAnalyticsReport,
    beta_risk: BetaRiskReport,
    peer_report: PeerStockAnalysisReport,
    debate_report: DebateReport,
    custom_plan: CustomAlgoPlan | None = None,
) -> tuple[pd.DataFrame, CustomAlgoReport]:
    """Build a data-justified hybrid schedule and simulate it."""
    if bars.empty:
        raise ValueError("Cannot build custom algo without execution-window bars.")

    fast_score = _fast_score(request, eda, beta_risk, peer_report, debate_report)
    liquidity_score = _liquidity_score(pretrade, peer_report, debate_report)
    if custom_plan is not None:
        fast_score = float(np.clip(0.55 * fast_score + 0.45 * custom_plan.urgency_score, 0.0, 1.0))
        liquidity_score = float(
            np.clip(0.55 * liquidity_score + 0.45 * custom_plan.liquidity_score, 0.0, 1.0)
        )
    weights = _component_weights(fast_score, liquidity_score)
    if custom_plan is not None and custom_plan.component_weights:
        weights = _merge_agent_component_weights(weights, custom_plan.component_weights)

    custom_name, style = _name_and_style(fast_score, liquidity_score, request)
    if custom_plan is not None and custom_plan.style_hint:
        custom_name, style = _agent_style_name(custom_plan.style_hint, custom_name, style)
    blend = _blended_curve(
        bars=bars,
        volume_curve=volume_curve,
        vwap_weight=weights["vwap_curve"],
        is_weight=weights["is_urgency"],
        pov_weight=weights["pov_guardrail"],
        twap_weight=weights["twap_stabilizer"],
        urgency=fast_score,
    )
    raw_child = _allocate_integer_quantity(request.quantity, blend).to_numpy(dtype=int)

    cap_rate = _adaptive_participation_cap(request, fast_score, liquidity_score, pretrade)
    if custom_plan is not None and custom_plan.max_participation_rate is not None:
        cap_rate = min(cap_rate, float(custom_plan.max_participation_rate))
    cap_rate = float(np.clip(cap_rate, 0.01, 0.50))
    capped_child, cap_violation_count, unallocated = _apply_participation_guardrail(
        raw_child=raw_child,
        bars=bars,
        parent_quantity=request.quantity,
        cap_rate=cap_rate,
        allow_force_complete=(
            request.pov_mode == "force_complete"
            or (custom_plan is not None and custom_plan.must_complete)
        )
        and not bool(custom_plan.strict_cap if custom_plan is not None else False),
    )
    completion_pct = custom_plan.completion_target_pct if custom_plan is not None else None
    completion_time = _parse_agent_time(custom_plan.completion_target_time) if custom_plan is not None else None
    capped_child, target_note = _apply_completion_target(
        child=capped_child,
        bars=bars,
        parent_quantity=request.quantity,
        cap_rate=cap_rate,
        target_pct=completion_pct,
        target_time=completion_time,
        allow_cap_violations=(
            custom_plan is not None
            and custom_plan.must_complete
            and not custom_plan.strict_cap
        ),
    )
    note = (
        f"{custom_name}: hybrid schedule from {weights['vwap_curve']:.0%} volume curve, "
        f"{weights['is_urgency']:.0%} IS urgency, {weights['pov_guardrail']:.0%} POV guardrail, "
        f"{weights['twap_stabilizer']:.0%} TWAP stabilizer. Adaptive cap {cap_rate:.1%}."
    )
    if target_note:
        note += f" {target_note}"
    if unallocated > 0:
        note += f" Strict-cap capacity left {unallocated:,} shares unallocated."

    schedule = pd.DataFrame(
        {
            "timestamp_et": list(bars["timestamp_et"]),
            "algo": "CUSTOM",
            "target_quantity": np.clip(capped_child, 0, None).astype(int),
            "parent_quantity": int(request.quantity),
            "bar_volume": bars["volume"].astype(float).to_numpy(),
            "participation_cap_quantity": np.floor(
                bars["volume"].astype(float).clip(lower=0).to_numpy() * cap_rate
            ).astype(int),
            "schedule_note": note,
        }
    )
    schedule["cap_violation"] = schedule["target_quantity"] > schedule["participation_cap_quantity"]
    cap_violation_count = int(schedule["cap_violation"].sum())
    if cap_violation_count > 0:
        note += f" Constraint handling created {cap_violation_count} cap-violation bars."
        schedule["schedule_note"] = note

    simulation = simulate_fills(
        algo="CUSTOM",
        schedule=schedule,
        bars=bars,
        side=request.side,
        spread_bps=request.spread_bps,
        impact_bps_per_10pct=request.impact_bps_per_10pct,
        limit_price=request.limit_price,
        pov_mode=request.pov_mode,
    )

    components = [
        CustomAlgoComponent(
            name="VWAP curve",
            weight=weights["vwap_curve"],
            reason=(
                f"Anchors the custom schedule to the 21-day liquidity shape; current volume is "
                f"{pretrade.current_vs_21d_volume:.2f}x the lookback baseline."
            ),
        ),
        CustomAlgoComponent(
            name="IS urgency",
            weight=weights["is_urgency"],
            reason=(
                f"Front-loads when adverse tape, beta risk, or peer crowding raises waiting risk; "
                f"fast score {fast_score:.2f}."
            ),
        ),
        CustomAlgoComponent(
            name="POV guardrail",
            weight=weights["pov_guardrail"],
            reason=(
                f"Caps footprint around {cap_rate:.1%} of displayed bar volume so impact stays visible."
            ),
        ),
        CustomAlgoComponent(
            name="TWAP stabilizer",
            weight=weights["twap_stabilizer"],
            reason="Keeps residual shares smooth when volume-curve estimates are noisy or sparse.",
        ),
    ]

    rationale = [
        (
            f"User brief: {request.custom_algo_instructions}"
            if request.custom_algo_instructions
            else "No custom desk brief was supplied, so the agent used market, risk, peer, and TCA context only."
        ),
        (
            f"CustomAlgoPlannerAgent objective: {custom_plan.objective_summary}"
            if custom_plan is not None
            else "CustomAlgoPlannerAgent did not return a plan; default market-context design was used."
        ),
        (
            f"21-day spread proxy is {pretrade.avg_spread_proxy_bps:.2f} bps and order size is "
            f"{pretrade.order_size_adv_pct * 100:.2f}% of ADV, so the design uses a footprint cap."
        ),
        (
            f"Timing risk map shows systematic {beta_risk.systematic_risk_bps:.2f} bps, sector "
            f"{beta_risk.sector_risk_bps:.2f} bps, and idiosyncratic {beta_risk.idiosyncratic_risk_bps:.2f} bps."
        ),
        (
            f"Peer agent reads '{peer_report.urgency_recommendation}' with crowding score "
            f"{peer_report.crowding_score:.2f}, which adjusts fast-vs-liquidity balance."
        ),
        (
            f"Debate judge recommended {debate_report.recommended_algo} after weighing "
            f"{debate_report.judge_winner}; custom algo borrows that stance without hiding caps."
        ),
    ]

    return schedule, CustomAlgoReport(
        name=custom_name,
        style=style,
        description=(
            "Agent-designed hybrid schedule that blends volume tracking, front-loaded urgency, "
            "displayed-volume participation guardrails, and a small TWAP stabilizer."
        ),
        components=components,
        parameters={
            "fast_score": round(fast_score, 4),
            "liquidity_score": round(liquidity_score, 4),
            "adaptive_participation_cap": round(cap_rate, 4),
            "unallocated_before_limits": int(unallocated),
            "cap_violation_count": int(cap_violation_count),
            "pov_mode": request.pov_mode,
            "limit_price": request.limit_price,
            "user_brief": request.custom_algo_instructions,
            "agent_plan": custom_plan.model_dump(mode="json") if custom_plan is not None else None,
            "agent_plan_status": custom_plan.status if custom_plan is not None else "unavailable",
            "agent_execution_story": custom_plan.execution_story if custom_plan is not None else "",
            "agent_operating_rules": custom_plan.operating_rules if custom_plan is not None else [],
        },
        rationale=rationale,
        simulation=simulation,
        caveats=[
            "Custom algo is generated from public OHLCV bars and agent-facing tool context, not venue-level microstructure.",
            "Custom desk briefs require CustomAlgoPlannerAgent/ADK for constraint interpretation; no hidden regex parser is used.",
            "Spread remains a high-low or user-entered proxy, not observed NBBO.",
            "The strategy is educational: use it to compare execution logic, not as production OMS routing advice.",
        ],
    )


def _fast_score(
    request: ExecutionRequest,
    eda: MarketEDA,
    beta_risk: BetaRiskReport,
    peer_report: PeerStockAnalysisReport,
    debate_report: DebateReport,
) -> float:
    score = 0.10 + 0.35 * float(request.urgency)
    adverse_tape = (request.side == "buy" and eda.price_move_bps > 0) or (
        request.side == "sell" and eda.price_move_bps < 0
    )
    if adverse_tape:
        score += 0.20
    total_factor = beta_risk.systematic_risk_bps + beta_risk.sector_risk_bps
    if total_factor > beta_risk.idiosyncratic_risk_bps:
        score += 0.15
    if peer_report.urgency_recommendation.startswith("Faster"):
        score += 0.15
    if debate_report.judge_winner == "FastExecutionAdvocate":
        score += 0.15
    return float(np.clip(score, 0.0, 1.0))


def _liquidity_score(
    pretrade: PreTradeAnalyticsReport,
    peer_report: PeerStockAnalysisReport,
    debate_report: DebateReport,
) -> float:
    score = 0.15
    if pretrade.avg_spread_proxy_bps >= 5.0:
        score += 0.25
    if pretrade.order_size_adv_pct >= 0.08:
        score += 0.20
    if pretrade.current_vs_21d_volume < 0.85:
        score += 0.15
    if peer_report.urgency_recommendation.startswith("Slower"):
        score += 0.15
    if debate_report.judge_winner == "LiquiditySeekingAdvocate":
        score += 0.15
    return float(np.clip(score, 0.0, 1.0))


def _component_weights(fast_score: float, liquidity_score: float) -> dict[str, float]:
    raw = {
        "vwap_curve": 0.30 + 0.20 * liquidity_score,
        "is_urgency": 0.20 + 0.35 * fast_score,
        "pov_guardrail": 0.25 + 0.20 * liquidity_score,
        "twap_stabilizer": max(0.08, 0.25 - 0.10 * max(fast_score, liquidity_score)),
    }
    total = sum(raw.values())
    return {key: float(value / total) for key, value in raw.items()}


def _merge_agent_component_weights(
    base_weights: dict[str, float],
    agent_weights: dict[str, float],
) -> dict[str, float]:
    allowed = {"vwap_curve", "is_urgency", "pov_guardrail", "twap_stabilizer"}
    cleaned = {
        key: float(np.clip(value, 0.0, 1.0))
        for key, value in agent_weights.items()
        if key in allowed and isinstance(value, int | float)
    }
    if not cleaned or sum(cleaned.values()) <= 0:
        return base_weights
    merged = {
        key: 0.45 * base_weights.get(key, 0.0) + 0.55 * cleaned.get(key, base_weights.get(key, 0.0))
        for key in allowed
    }
    total = sum(merged.values())
    return {key: float(value / total) for key, value in merged.items()}


def _name_and_style(
    fast_score: float,
    liquidity_score: float,
    request: ExecutionRequest,
) -> tuple[str, str]:
    if fast_score - liquidity_score >= 0.18:
        return "Apex IS-VWAP Hybrid", "Front-loaded liquidity-aware implementation shortfall"
    if liquidity_score - fast_score >= 0.18:
        return "Quiet Liquidity Seeker", "VWAP/POV schedule with strict footprint discipline"
    if request.limit_price is not None:
        return "Limit-Aware Adaptive VWAP", "Balanced VWAP with limit feasibility guardrails"
    return "Balanced Adaptive VWAP", "Blended volume tracking with moderate timing protection"


def _agent_style_name(style_hint: str, fallback_name: str, fallback_style: str) -> tuple[str, str]:
    normalized = str(style_hint or "").strip().lower().replace("-", "_").replace(" ", "_")
    if "liquidity" in normalized:
        return "Agent Liquidity Seeker", "Agent-planned VWAP/POV liquidity-seeking hybrid"
    if "front" in normalized or "is" in normalized or "urgent" in normalized:
        return "Agent Front-Loaded IS", "Agent-planned front-loaded implementation shortfall hybrid"
    if "limit" in normalized:
        return "Agent Limit-Aware VWAP", "Agent-planned VWAP schedule with limit and completion guardrails"
    if "pov" in normalized:
        return "Agent Capped POV Hybrid", "Agent-planned participation-capped adaptive schedule"
    if "adaptive" in normalized or "hybrid" in normalized:
        return "Agent Adaptive Hybrid", "Agent-planned blend of urgency, volume, and participation controls"
    return fallback_name, fallback_style


def _parse_agent_time(raw: str | None) -> time | None:
    if not raw:
        return None
    text = str(raw).strip().lower()
    try:
        parts = text.replace("am", "").replace("pm", "").split(":")
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
    except Exception:
        return None
    if "pm" in text and hour < 12:
        hour += 12
    if "am" in text and hour == 12:
        hour = 0
    if "am" not in text and "pm" not in text and hour < 9:
        hour += 12
    return time(hour=max(0, min(23, hour)), minute=max(0, min(59, minute)))


def _blended_curve(
    bars: pd.DataFrame,
    volume_curve: pd.Series,
    vwap_weight: float,
    is_weight: float,
    pov_weight: float,
    twap_weight: float,
    urgency: float,
) -> pd.Series:
    index = bars.index
    n = len(bars)
    vwap = volume_curve.reindex(index).fillna(0).astype(float)
    if float(vwap.sum()) <= 0:
        vwap = bars["volume"].astype(float).clip(lower=0)
    vwap = _normalize(vwap)

    x = np.linspace(0.0, 1.0, n)
    is_curve = pd.Series(np.exp(-(1.0 + 5.0 * urgency) * x), index=index)
    is_curve = _normalize(is_curve)

    pov = _normalize(bars["volume"].astype(float).clip(lower=0))
    twap = pd.Series(np.ones(n), index=index)
    twap = _normalize(twap)

    blend = vwap_weight * vwap + is_weight * is_curve + pov_weight * pov + twap_weight * twap
    return _normalize(blend)


def _adaptive_participation_cap(
    request: ExecutionRequest,
    fast_score: float,
    liquidity_score: float,
    pretrade: PreTradeAnalyticsReport,
) -> float:
    base = max(float(request.participation_rate), 0.08 + 0.12 * fast_score)
    if pretrade.order_size_adv_pct > 0.10:
        base += 0.04
    if liquidity_score > fast_score:
        base -= 0.03 * (liquidity_score - fast_score)
    return float(np.clip(base, request.participation_rate, 0.35))


def _apply_participation_guardrail(
    raw_child: np.ndarray,
    bars: pd.DataFrame,
    parent_quantity: int,
    cap_rate: float,
    allow_force_complete: bool,
) -> tuple[np.ndarray, int, int]:
    caps = np.floor(bars["volume"].astype(float).clip(lower=0).to_numpy() * cap_rate).astype(int)
    child = np.minimum(np.clip(raw_child.astype(int), 0, None), caps)
    remaining = int(parent_quantity) - int(child.sum())

    loops = 0
    while remaining > 0 and loops < 4:
        capacity = np.clip(caps - child, 0, None)
        if int(capacity.sum()) <= 0:
            break
        extra = _allocate_integer_quantity(remaining, pd.Series(capacity)).to_numpy(dtype=int)
        extra = np.minimum(extra, capacity)
        if int(extra.sum()) <= 0:
            break
        child += extra
        remaining = int(parent_quantity) - int(child.sum())
        loops += 1

    cap_violation_count = 0
    if remaining > 0 and allow_force_complete:
        weights = pd.Series(bars["volume"].astype(float).clip(lower=0).to_numpy())
        catch_up = _allocate_integer_quantity(remaining, weights).to_numpy(dtype=int)
        child += catch_up
        remaining = 0
        cap_violation_count = int((child > caps).sum())

    return child.astype(int), cap_violation_count, max(0, remaining)


def _apply_completion_target(
    child: np.ndarray,
    bars: pd.DataFrame,
    parent_quantity: int,
    cap_rate: float,
    target_pct: object,
    target_time: object,
    allow_cap_violations: bool,
) -> tuple[np.ndarray, str]:
    if target_pct is None or target_time is None:
        return child.astype(int), ""
    if not isinstance(target_time, time):
        return child.astype(int), ""

    adjusted = np.clip(child.astype(int), 0, None)
    timestamps = pd.to_datetime(bars["timestamp_et"])
    before_deadline = np.array([stamp.time() <= target_time for stamp in timestamps])
    after_deadline = ~before_deadline
    if not bool(before_deadline.any()) or not bool(after_deadline.any()):
        return adjusted, ""

    target_shares = int(np.ceil(float(target_pct) * int(parent_quantity)))
    current_before = int(adjusted[before_deadline].sum())
    deficit = max(0, target_shares - current_before)
    if deficit <= 0:
        return adjusted, (
            f"User target already satisfied: {target_pct:.0%} scheduled by {target_time.strftime('%H:%M')}."
        )

    after_indices = np.where(after_deadline & (adjusted > 0))[0][::-1]
    before_indices = np.where(before_deadline)[0]
    caps = np.floor(bars["volume"].astype(float).clip(lower=0).to_numpy() * cap_rate).astype(int)
    capacity = np.clip(caps - adjusted, 0, None)

    moved = 0
    for source in after_indices:
        if deficit <= 0:
            break
        available = int(adjusted[source])
        if available <= 0:
            continue
        take = min(available, deficit)
        adjusted[source] -= take
        remaining_take = take
        weights = pd.Series(capacity[before_indices], index=before_indices)
        if int(weights.sum()) > 0:
            allocations = _allocate_integer_quantity(remaining_take, weights)
            for dest, add in allocations.items():
                add_int = min(int(add), int(capacity[int(dest)]))
                adjusted[int(dest)] += add_int
                capacity[int(dest)] -= add_int
                remaining_take -= add_int
                moved += add_int
                deficit -= add_int
                if remaining_take <= 0 or deficit <= 0:
                    break
        if remaining_take > 0 and allow_cap_violations:
            weights = pd.Series(bars.iloc[before_indices]["volume"].astype(float).to_numpy(), index=before_indices)
            allocations = _allocate_integer_quantity(remaining_take, weights)
            for dest, add in allocations.items():
                add_int = int(add)
                adjusted[int(dest)] += add_int
                moved += add_int
                deficit -= add_int
        elif remaining_take > 0:
            adjusted[source] += remaining_take

    if moved <= 0:
        return adjusted, (
            f"User target {target_pct:.0%} by {target_time.strftime('%H:%M')} could not be met under the cap."
        )
    remaining_gap = max(0, target_shares - int(adjusted[before_deadline].sum()))
    if remaining_gap > 0:
        return adjusted, (
            f"Shifted {moved:,} shares earlier toward {target_pct:.0%} by {target_time.strftime('%H:%M')}, "
            f"but {remaining_gap:,} shares remain short under constraints."
        )
    return adjusted, f"Shifted {moved:,} shares earlier to target {target_pct:.0%} by {target_time.strftime('%H:%M')}."


def _serializable_constraints(constraints: dict[str, object]) -> dict[str, object]:
    payload = dict(constraints)
    if isinstance(payload.get("completion_time"), time):
        payload["completion_time"] = payload["completion_time"].strftime("%H:%M")
    return payload


def _normalize(values: pd.Series) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce").fillna(0).clip(lower=0)
    if len(clean) == 0:
        return clean.astype(float)
    if float(clean.sum()) <= 0:
        clean = pd.Series(np.ones(len(clean)), index=clean.index)
    return clean / float(clean.sum())


def _allocate_integer_quantity(quantity: int, weights: pd.Series) -> pd.Series:
    clean = _normalize(weights)
    if len(clean) == 0:
        return clean.astype(int)
    raw = clean * int(quantity)
    base = np.floor(raw).astype(int)
    remainder = int(quantity) - int(base.sum())
    if remainder > 0:
        fractional_order = np.argsort(-(raw - base).to_numpy())
        for idx in fractional_order[:remainder]:
            base.iloc[idx] += 1
    elif remainder < 0:
        reduce_order = np.argsort((raw - base).to_numpy())
        for idx in reduce_order[: abs(remainder)]:
            if base.iloc[idx] > 0:
                base.iloc[idx] -= 1
    return base.astype(int)
