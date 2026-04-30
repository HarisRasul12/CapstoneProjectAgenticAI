from __future__ import annotations

from execlab.config import Settings
from execlab.schemas import AgentStepReport, ExecutionMemo


def adk_is_available() -> bool:
    try:
        import google.adk.agents  # noqa: F401
        import google.adk.runners  # noqa: F401
        import google.adk.sessions  # noqa: F401
        import google.genai.types  # noqa: F401
    except Exception:
        return False
    return True


def create_execlab_root_agent(settings: Settings, model_name: str | None = None):
    from google.adk.agents import LlmAgent, SequentialAgent

    model = model_name or settings.vertex_model

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
        description="Runs bar-based fill simulation.",
        instruction=(
            "Role: deterministic execution simulation controller.\n"
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
            "and beta/systematic-vs-idiosyncratic risk. Use only provided numbers."
        ),
        tools=[],
        output_schema=AgentStepReport,
        output_key="cause_effect_report",
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
            "{simulation_report}, {tca_report}, {cause_effect_report}, {limit_feasibility_report}.\n"
            "Few-shot style target:\n"
            "Example 1: 'The stock moved upward after the open, so the front-loaded IS schedule "
            "benefited from trading earlier. VWAP stayed closest to market VWAP because it tracked "
            "the volume curve. TWAP lagged because it kept trading into higher prices.'\n"
            "Example 2: 'On a falling tape, slower schedules can look better versus arrival, but "
            "that does not mean they controlled risk; it reflects the realized price path.'\n"
            "Return ExecutionMemo. Include a clear recommendation on algo choice, spread/impact "
            "settings, beta/systematic risk versus idiosyncratic timing risk, strict POV/limit "
            "behavior, and the OMS/EMS limitation exactly once."
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
            "prefer TWAP, VWAP, POV, or IS and explain the spread, impact, market beta, "
            "and idiosyncratic timing-risk tradeoff."
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
            limit_agent,
            narrative_agent,
            critic_agent,
        ],
    )
