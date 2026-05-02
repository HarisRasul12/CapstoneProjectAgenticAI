from __future__ import annotations

from datetime import date, time

import pandas as pd

from execlab.analytics import build_market_eda
from execlab.custom_algo import build_custom_algo_report
from execlab.pretrade import build_pretrade_analytics, fetch_lookback_sessions
from execlab.risk import build_beta_risk_report, build_peer_stock_analysis
from execlab.schemas import CustomAlgoPlan, DebateCase, DebateReport, ExecutionRequest
from tests.conftest import FakeMarketDataClient, synthetic_bars


def test_custom_algo_uses_agent_plan_for_completion_and_participation() -> None:
    request, bars, volume_curve, eda, pretrade, beta, peer, debate = _custom_inputs()
    plan = CustomAlgoPlan(
        status="ok",
        objective_summary="PM wants front-loaded exposure reduction with explicit cap discipline.",
        urgency_score=0.90,
        liquidity_score=0.70,
        max_participation_rate=0.12,
        completion_target_pct=0.60,
        completion_target_time="11:00",
        must_complete=False,
        strict_cap=True,
        pm_exposure_summary="Reduce exposure before the late-morning risk window.",
        risk_constraints=["Keep participation at or below 12%."],
        style_hint="front_loaded_is",
        execution_story=(
            "This custom algo starts with controlled urgency because the PM wants exposure down "
            "before 11:00. It leans on IS-style front-loading early, but it does not simply sweep "
            "the tape; the POV guardrail keeps child orders near the requested cap. After the "
            "deadline, it settles into volume-aware pacing and uses a small TWAP stabilizer so "
            "residual shares do not become lumpy. The trader should expect a fast first act, then "
            "a more patient liquidity-seeking finish."
        ),
        operating_rules=[
            "Prioritize the 11:00 completion target without breaching the 12% cap.",
            "Use volume curve liquidity before forcing urgency.",
            "Escalate for desk review if cap capacity cannot meet the target.",
        ],
        component_weights={
            "vwap_curve": 0.20,
            "is_urgency": 0.45,
            "pov_guardrail": 0.25,
            "twap_stabilizer": 0.10,
        },
        rationale=["Front-load because the user specified a PM exposure deadline."],
    )

    schedule, report = build_custom_algo_report(
        request=request,
        bars=bars,
        volume_curve=volume_curve,
        eda=eda,
        pretrade=pretrade,
        beta_risk=beta,
        peer_report=peer,
        debate_report=debate,
        custom_plan=plan,
    )

    fills_by_deadline = schedule.loc[
        pd.to_datetime(schedule["timestamp_et"]).dt.time <= time(11, 0),
        "target_quantity",
    ].sum()
    max_participation = (schedule["target_quantity"] / schedule["bar_volume"]).max()

    assert report.parameters["agent_plan"]["objective_summary"].startswith("PM wants")
    assert report.parameters["agent_plan_status"] == "ok"
    assert report.parameters["agent_execution_story"].startswith("This custom algo starts")
    assert report.parameters["agent_operating_rules"]
    assert fills_by_deadline >= int(0.60 * request.quantity)
    assert max_participation <= 0.12001
    assert report.components


def test_custom_algo_does_not_parse_chat_brief_without_agent_plan() -> None:
    request, bars, volume_curve, eda, pretrade, beta, peer, debate = _custom_inputs(
        brief="PM wants 60% done by 11:00 and max participation 12%."
    )

    schedule, report = build_custom_algo_report(
        request=request,
        bars=bars,
        volume_curve=volume_curve,
        eda=eda,
        pretrade=pretrade,
        beta_risk=beta,
        peer_report=peer,
        debate_report=debate,
        custom_plan=None,
    )

    assert report.parameters["agent_plan"] is None
    assert report.parameters["agent_plan_status"] == "unavailable"
    assert report.parameters["agent_execution_story"] == ""
    assert report.parameters["agent_operating_rules"] == []
    assert "interpreted_constraints" not in report.parameters
    assert schedule["target_quantity"].sum() == request.quantity


def _custom_inputs(brief: str = ""):
    client = FakeMarketDataClient()
    request = ExecutionRequest(
        ticker="NVDA",
        trade_date=date(2026, 4, 20),
        quantity=25_000,
        custom_algo_instructions=brief,
    )
    bars = synthetic_bars()
    volume_curve = bars["volume"].astype(float) / float(bars["volume"].sum())
    eda = build_market_eda(
        ticker=request.ticker,
        trade_date=request.trade_date,
        interval=request.interval,
        all_bars=bars,
        window_bars=bars,
        volume_curve_source="test actual curve",
    )
    lookback, warnings = fetch_lookback_sessions(client, request.ticker, request.trade_date, "5m", 5)
    pretrade = build_pretrade_analytics(request, bars, lookback, requested_sessions=5, warnings=warnings)
    beta = build_beta_risk_report(request, client, lookback_days=63)
    peer = build_peer_stock_analysis(request, client, beta, lookback_days=63, max_candidates=6)
    debate = DebateReport(
        fast_case=DebateCase(
            advocate="FastExecutionAdvocate",
            stance="Reduce timing risk.",
            recommended_algos=["IS"],
            thesis="Use urgency when exposure matters.",
        ),
        liquidity_case=DebateCase(
            advocate="LiquiditySeekingAdvocate",
            stance="Control footprint.",
            recommended_algos=["VWAP"],
            thesis="Use liquidity when impact matters.",
        ),
        judge_winner="FastExecutionAdvocate",
        recommended_algo="IS",
        confidence=0.74,
        deciding_factors=["Adverse timing risk and PM exposure objective."],
        judge_rationale="Fast execution is better supported for this mocked custom brief.",
    )
    return request, bars, volume_curve, eda, pretrade, beta, peer, debate
