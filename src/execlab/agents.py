from __future__ import annotations

from pathlib import Path

from execlab.config import Settings
from execlab.schemas import AgentStepReport, CustomAlgoPlan, ExecutionMemo


def adk_is_available() -> bool:
    try:
        import google.adk.agents  # noqa: F401
        import google.adk.runners  # noqa: F401
        import google.adk.sessions  # noqa: F401
        import google.genai.types  # noqa: F401
    except Exception:
        return False
    return True


def _load_few_shot_guidance() -> str:
    path = Path(__file__).resolve().parent / "prompt" / "few_shots.yaml"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return (
            "Use visible analyst rationale: Observation -> Driver -> Effect -> Recommendation. "
            "Do not expose hidden chain-of-thought."
        )
    # Keep the prompt bounded for latency while preserving the full style guide and many examples.
    return text[:18_000]


def create_execlab_root_agent(settings: Settings, model_name: str | None = None):
    from google.adk.agents import LlmAgent, SequentialAgent

    model = model_name or settings.vertex_model
    few_shot_guidance = _load_few_shot_guidance()

    market_agent = LlmAgent(
        name="MarketDataAgent",
        model=model,
        description="Summarizes already-computed live intraday OHLCV data.",
        instruction=(
            "Role: market data analyst for an execution backtest.\n"
            "Input: {execution_context}.\n"
            "Summarize data availability, regular-session coverage, arrival price, market VWAP, "
            "close price, price move, and any warnings. Use only provided context."
        ),
        tools=[],
        output_schema=AgentStepReport,
        output_key="market_data_report",
    )

    volume_agent = LlmAgent(
        name="VolumeCurveAgent",
        model=model,
        description="Builds and explains the volume curve used by VWAP.",
        instruction=(
            "Role: intraday volume-curve analyst.\n"
            "Input: {execution_context}.\n"
            "Explain the 21-day volume curve source, current-vs-historical volume, and how VWAP "
            "uses the curve. Use only provided context."
        ),
        tools=[],
        output_schema=AgentStepReport,
        output_key="volume_curve_report",
    )

    pretrade_agent = LlmAgent(
        name="PreTradeAnalyticsAgent",
        model=model,
        description="Summarizes 21-day liquidity, spread-proxy, volatility, and time-risk curves.",
        instruction=(
            "Role: pre-trade liquidity analyst.\n"
            "Input: {execution_context}.\n"
            "Explain ADV, order-size-vs-ADV, 21-day volume curve, high-low spread proxy, "
            "volatility curve, and time risk. Clearly state that spread is an OHLC high-low "
            "proxy, not NBBO."
        ),
        tools=[],
        output_schema=AgentStepReport,
        output_key="pretrade_report_agent",
    )

    expected_cost_agent = LlmAgent(
        name="ExpectedCostModelAgent",
        model=model,
        description="Explains the educational expected-cost regression and cost breakdown.",
        instruction=(
            "Role: transparent expected-cost model analyst.\n"
            "Input: {execution_context}.\n"
            "Explain expected cost, regression quality, and spread/impact/timing/drift/limit components. "
            "Recommend practical spread and participation settings when the data supports it. "
            "Do not overstate model precision."
        ),
        tools=[],
        output_schema=AgentStepReport,
        output_key="expected_cost_report_agent",
    )

    historical_regression_agent = LlmAgent(
        name="HistoricalRegressionAgent",
        model=model,
        description="Audits the historical feature engineering and expected-cost regression fit.",
        instruction=(
            "Role: historical regression analyst.\n"
            "Input: {execution_context}.\n"
            "Explain how the 21-day intraday rows, participation, spread proxy, volatility, "
            "time risk, relative volume, order-size-vs-ADV, and drift proxy feed the expected-cost "
            "regression. State observation count, R-squared, strongest coefficient signs, and caveats."
        ),
        tools=[],
        output_schema=AgentStepReport,
        output_key="historical_regression_report",
    )

    beta_risk_agent = LlmAgent(
        name="BetaRiskMappingAgent",
        model=model,
        description="Maps the ticker to market/sector ETFs and explains systematic versus idiosyncratic risk.",
        instruction=(
            "Role: market-risk mapper for execution timing.\n"
            "Input: {execution_context}.\n"
            "Explain the selected market ETF and sector ETF, mapping confidence, market beta, sector beta, "
            "correlations, systematic risk bps, sector risk bps, and idiosyncratic risk bps. "
            "Make the distinction between impact cost and market timing risk explicit."
        ),
        tools=[],
        output_schema=AgentStepReport,
        output_key="beta_risk_report_agent",
    )

    peer_cluster_agent = LlmAgent(
        name="PeerClusterAgent",
        model=model,
        description="Finds closest peer stocks and explains correlation clusters and urgency implications.",
        instruction=(
            "Role: peer flow and clustering analyst.\n"
            "Input: {execution_context}.\n"
            "Explain the closest peers, correlations, recent peer moves, crowding score, and whether "
            "peer behavior argues for faster execution, slower liquidity-seeking, or balanced VWAP/POV. "
            "Distinguish peer-driven market impact pressure from stock-specific idiosyncratic risk."
        ),
        tools=[],
        output_schema=AgentStepReport,
        output_key="peer_cluster_report_agent",
    )

    strategy_agent = LlmAgent(
        name="AlgoStrategyAgent",
        model=model,
        description="Generates benchmark execution schedules.",
        instruction=(
            "Role: execution strategy analyst.\n"
            "Input: {execution_context}.\n"
            "Compare TWAP, VWAP, POV, and IS schedule behavior. Highlight fill timing, "
            "completion rate, cap violations, and whether strict POV leaves unfilled quantity."
        ),
        tools=[],
        output_schema=AgentStepReport,
        output_key="strategy_report",
    )

    simulator_agent = LlmAgent(
        name="ExecutionSimulatorAgent",
        model=model,
        description="Explains bar-based fill simulation.",
        instruction=(
            "Role: tool-grounded execution simulation analyst.\n"
            "Input: {execution_context}.\n"
            "Summarize modeled fills, unfilled quantities, limit-price blocks, and completion rates. "
            "Use only provided context."
        ),
        tools=[],
        output_schema=AgentStepReport,
        output_key="simulation_report",
    )

    benchmark_agent = LlmAgent(
        name="BenchmarkTcaAgent",
        model=model,
        description="Computes arrival, VWAP, and close benchmark slippage.",
        instruction=(
            "Role: transaction cost analysis analyst.\n"
            "Input: {execution_context}.\n"
            "Identify best and worst algorithms using positive cost as worse. Compare arrival, "
            "VWAP, close, and scenario expected costs."
        ),
        tools=[],
        output_schema=AgentStepReport,
        output_key="tca_report",
    )

    cause_effect_agent = LlmAgent(
        name="CauseEffectTcaAgent",
        model=model,
        description="Turns numeric TCA outputs into cause-and-effect execution bullets.",
        instruction=(
            "Role: cause-and-effect TCA analyst.\n"
            "Input: {execution_context}.\n"
            "Write precise bullets explaining why one algo beat another: price-path timing, "
            "volume-curve fit, participation cap, spread/volatility friction, limit feasibility, "
            "and beta/systematic-vs-idiosyncratic risk. Use only provided numbers.\n"
            "Use this public reasoning format from the prompt asset. Do not expose hidden chain-of-thought:\n"
            f"{few_shot_guidance[:5000]}"
        ),
        tools=[],
        output_schema=AgentStepReport,
        output_key="cause_effect_report",
    )

    fast_advocate_agent = LlmAgent(
        name="FastExecutionAdvocate",
        model=model,
        description="Argues for front-loaded execution when timing risk dominates liquidity cost.",
        instruction=(
            "Role: execution debate advocate favoring faster execution.\n"
            "Input: {execution_context}.\n"
            "Make the strongest numeric case for IS/front-loaded or higher-urgency execution using "
            "arrival cost, adverse price path, beta risk, peer crowding, and completion urgency. "
            "Also state the spread/impact caveat."
        ),
        tools=[],
        output_schema=AgentStepReport,
        output_key="fast_execution_argument",
    )

    liquidity_advocate_agent = LlmAgent(
        name="LiquiditySeekingAdvocate",
        model=model,
        description="Argues for VWAP/TWAP/POV when footprint control dominates timing risk.",
        instruction=(
            "Role: execution debate advocate favoring liquidity-seeking execution.\n"
            "Input: {execution_context}.\n"
            "Make the strongest numeric case for VWAP/TWAP/strict POV using spread proxy, "
            "participation, volume curve fit, completion, and limit feasibility. "
            "Also state the timing-risk caveat."
        ),
        tools=[],
        output_schema=AgentStepReport,
        output_key="liquidity_seeking_argument",
    )

    debate_judge_agent = LlmAgent(
        name="DebateJudgeAgent",
        model=model,
        description="Judges the fast-vs-liquidity debate using tool-grounded report outputs.",
        instruction=(
            "Role: neutral execution debate judge.\n"
            "Inputs: {execution_context}, {fast_execution_argument}, {liquidity_seeking_argument}.\n"
            "Choose the stronger argument using the agent_debate report. Explain why "
            "the recommended algo is robust or fragile. Use concise visible rationale only."
        ),
        tools=[],
        output_schema=AgentStepReport,
        output_key="debate_judge_report",
    )

    counterfactual_agent = LlmAgent(
        name="CounterfactualAgent",
        model=model,
        description="Explains what assumptions would change the winning execution algorithm.",
        instruction=(
            "Role: counterfactual TCA analyst.\n"
            "Input: {execution_context}.\n"
            "Explain the counterfactual scenarios: flat tape, wider spread, larger order, adverse "
            "peer crowding, and completion-adjusted view. State which assumptions change the winner "
            "and what that means for robustness."
        ),
        tools=[],
        output_schema=AgentStepReport,
        output_key="counterfactual_report_agent",
    )

    playbook_agent = LlmAgent(
        name="ExecutionPlaybookAgent",
        model=model,
        description="Turns the debate and counterfactual analysis into a desk-style execution plan.",
        instruction=(
            "Role: execution playbook writer.\n"
            "Input: {execution_context}.\n"
            "Summarize recommended algo, urgency, participation guidance, limit guidance, monitoring "
            "triggers, and switch rules. Keep the guidance practical and caveated."
        ),
        tools=[],
        output_schema=AgentStepReport,
        output_key="playbook_report_agent",
    )

    custom_algo_agent = LlmAgent(
        name="CustomAlgoDesignerAgent",
        model=model,
        description="Explains the generated custom hybrid algorithm and why its components fit the tape.",
        instruction=(
            "Role: custom execution algorithm designer.\n"
            "Input: {execution_context}.\n"
            "Read custom_algo_agent_plan as the agent-authored translation of the user's desk brief. "
            "Explain how the design responds to urgency, max participation, completion-by-time target, "
            "PM exposure, risk constraints, and limit guidance when those are present. Explain the "
            "custom_algo section: name, style, component weights, adaptive participation cap, modeled arrival cost, "
            "completion, cap violations, and why this hybrid would fit the selected volume curve, "
            "spread proxy, beta risk, peer crowding, and debate result. Keep it clear that "
            "this is an educational bar-based strategy, not production routing. Return status='ok' and "
            "3-5 highlights using visible analyst reasoning: observation, driver, implication."
        ),
        tools=[],
        output_schema=AgentStepReport,
        output_key="custom_algo_designer_report",
    )

    tab_insight_agent = LlmAgent(
        name="TabInsightAgent",
        model=model,
        description="Summarizes the data-driven insight cards rendered in every Streamlit tab.",
        instruction=(
            "Role: dashboard insight narrator.\n"
            "Input: {execution_context}.\n"
            "Read the market, pretrade, expected_cost_model, beta_risk, peer_analysis, causal_tca, "
            "agent_debate, custom_algo, algo_metrics, and scenario_results sections directly. "
            "Do not rely on any prewritten insight report. Generate fresh dashboard commentary for "
            "Pre-Trade Lab, Risk Model, Peers, Custom Algo, regular TCA, Scenario Lab, and Agent Memo. "
            "Use concise visible analyst rationale only. Return status='ok' and 3-5 highlights that "
            "explain why the tabs matter for an execution decision."
        ),
        tools=[],
        output_schema=AgentStepReport,
        output_key="tab_insight_report_agent",
    )

    limit_agent = LlmAgent(
        name="LimitFeasibilityAgent",
        model=model,
        description="Explains limit-price feasibility and unfilled risk.",
        instruction=(
            "Role: limit-order feasibility analyst.\n"
            "Input: {execution_context}.\n"
            "Explain what can execute, what is blocked by limit price, and how unfilled quantity "
            "changes algorithm comparison. If no limit was used, state that explicitly."
        ),
        tools=[],
        output_schema=AgentStepReport,
        output_key="limit_feasibility_report",
    )

    narrative_agent = LlmAgent(
        name="NarrativeExplanationAgent",
        model=model,
        description="Writes an execution analyst memo using few-shot style guidance.",
        instruction=(
            "Role: execution analyst writing a concise TCA memo.\n"
            "Inputs: {execution_context}, {market_data_report}, {volume_curve_report}, "
            "{pretrade_report_agent}, {expected_cost_report_agent}, {historical_regression_report}, "
            "{beta_risk_report_agent}, {peer_cluster_report_agent}, {strategy_report}, "
            "{simulation_report}, {tca_report}, {cause_effect_report}, {fast_execution_argument}, "
            "{liquidity_seeking_argument}, {debate_judge_report}, {counterfactual_report_agent}, "
            "{playbook_report_agent}, {custom_algo_designer_report}, {tab_insight_report_agent}, "
            "{limit_feasibility_report}.\n"
            "Few-shot guidance and public reasoning examples:\n"
            f"{few_shot_guidance}\n"
            "Return ExecutionMemo. Include a clear recommendation on algo choice, spread/impact "
            "settings, beta/systematic risk versus idiosyncratic timing risk, strict POV/limit "
            "behavior, whether the custom hybrid improves the desk recommendation, and the "
            "OMS/EMS limitation exactly once. Use concise visible rationale, "
            "not hidden chain-of-thought."
        ),
        tools=[],
        output_schema=ExecutionMemo,
        output_key="execution_memo_draft",
    )

    critic_agent = LlmAgent(
        name="CriticGoldenSetAgent",
        model=model,
        description="Refines the memo for benchmark math, caveats, and rubric readiness.",
        instruction=(
            "Role: skeptical execution-review critic.\n"
            "Inputs: {execution_context} and {execution_memo_draft}.\n"
            "Return a final ExecutionMemo that is numeric, plain-English, explicitly caveated, "
            "and directly answers how to execute. Do not invent values. Recommend whether to "
            "prefer TWAP, VWAP, POV, IS, or the custom hybrid as a research idea, and explain "
            "the spread, impact, market beta, and idiosyncratic timing-risk tradeoff."
        ),
        tools=[],
        output_schema=ExecutionMemo,
        output_key="execution_memo",
    )

    return SequentialAgent(
        name="ExecLabCoordinatorAgent",
        sub_agents=[
            market_agent,
            volume_agent,
            pretrade_agent,
            expected_cost_agent,
            historical_regression_agent,
            beta_risk_agent,
            peer_cluster_agent,
            strategy_agent,
            simulator_agent,
            benchmark_agent,
            cause_effect_agent,
            fast_advocate_agent,
            liquidity_advocate_agent,
            debate_judge_agent,
            counterfactual_agent,
            playbook_agent,
            custom_algo_agent,
            tab_insight_agent,
            limit_agent,
            narrative_agent,
            critic_agent,
        ],
    )


def create_custom_algo_planner_agent(settings: Settings, model_name: str | None = None):
    from google.adk.agents import LlmAgent

    model = model_name or settings.vertex_model
    return LlmAgent(
        name="CustomAlgoPlannerAgent",
        model=model,
        description="Turns a trader's custom algo chat brief into a structured execution plan.",
        instruction=(
            "Role: custom execution algorithm planner.\n"
            "Inputs: {request}, {custom_planner_context}.\n"
            "Interpret the user's desk brief as execution constraints. Extract only what is "
            "supported by the user's words and the provided market context. Return CustomAlgoPlan.\n"
            "Guidelines:\n"
            "- If the user specifies a max participation cap, set max_participation_rate as a decimal.\n"
            "- If the user says a percent must be done by a time, set completion_target_pct and "
            "completion_target_time in HH:MM 24-hour market time.\n"
            "- Use urgency_score for PM exposure, adverse tape, must-complete, or risk-reduction needs.\n"
            "- Use liquidity_score for minimize-impact, low-footprint, no-chase, or strict-cap needs.\n"
            "- component_weights may include vwap_curve, is_urgency, pov_guardrail, twap_stabilizer; "
            "weights should be positive and roughly sum to 1.\n"
            "- If a brief is ambiguous, still provide a usable educational plan and put clarifying "
            "items in follow_up_questions.\n"
            "- Do not reveal hidden chain-of-thought; rationale should be concise visible analyst bullets."
        ),
        tools=[],
        output_schema=CustomAlgoPlan,
        output_key="custom_algo_plan",
    )
