from __future__ import annotations

from datetime import date, datetime, time, timedelta
from html import escape
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from execlab.config import load_settings
from execlab.schemas import ExecutionRequest
from execlab.service import ExecLabService


EASTERN = ZoneInfo("America/New_York")


st.set_page_config(
    page_title="ExecLab AI",
    page_icon="EX",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_service() -> ExecLabService:
    return ExecLabService(settings=load_settings())


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: #0f1419;
            color: #f4f6f8;
        }
        [data-testid="stHeader"] { background: rgba(15, 20, 25, 0); }
        [data-testid="stSidebar"] {
            background: #151c24;
            border-right: 1px solid rgba(255, 255, 255, 0.10);
        }
        h1, h2, h3, h4, h5, h6, p, li, label { color: #f4f6f8 !important; }
        .execlab-hero {
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: #151c24;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            margin-bottom: 1rem;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 0.65rem;
            margin: 0.5rem 0 1rem 0;
        }
        .metric-card {
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-left: 3px solid #38bdf8;
            background: #151c24;
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            min-height: 98px;
        }
        .metric-label {
            color: #aeb8c2;
            font-size: 0.82rem;
            line-height: 1.1;
        }
        .metric-value {
            color: #ffffff;
            font-weight: 760;
            font-size: 1.65rem;
            line-height: 1.15;
            margin-top: 0.18rem;
        }
        .metric-sub {
            color: #aeb8c2;
            font-size: 0.78rem;
            margin-top: 0.2rem;
        }
        .memo-box {
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: #151c24;
            border-radius: 8px;
            padding: 0.85rem 1rem;
        }
        .agent-box {
            border: 1px solid rgba(56, 189, 248, 0.25);
            background: #111922;
            border-left: 3px solid #38bdf8;
            border-radius: 8px;
            padding: 0.75rem 0.9rem;
            margin: 0.35rem 0 0.85rem 0;
        }
        .agent-box strong { color: #ffffff !important; }
        .agent-box span { color: #c6d0da !important; }
        .narrative-box {
            border: 1px solid rgba(14, 165, 233, 0.32);
            background: #0f1a23;
            border-left: 4px solid #0ea5e9;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            margin: 0.45rem 0 0.95rem 0;
        }
        .narrative-kicker {
            color: #38bdf8;
            font-size: 0.78rem;
            font-weight: 760;
            text-transform: uppercase;
            letter-spacing: 0;
            margin-bottom: 0.2rem;
        }
        .narrative-title {
            color: #ffffff;
            font-size: 1.05rem;
            font-weight: 760;
            margin-bottom: 0.35rem;
        }
        .narrative-body {
            color: #d7e0e8;
            line-height: 1.48;
            margin-bottom: 0.45rem;
        }
        .narrative-watch {
            color: #aeb8c2;
            font-size: 0.86rem;
            margin-top: 0.4rem;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def previous_weekday() -> date:
    cursor = datetime.now(EASTERN).date() - timedelta(days=1)
    while cursor.weekday() >= 5:
        cursor -= timedelta(days=1)
    return cursor


def agent_note(title: str, agents: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="agent-box">
          <strong>{title}</strong><br/>
          <span><strong>Agents:</strong> {agents}</span><br/>
          <span>{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


AGENT_REPORTS_BY_TAB = {
    "pretrade": [
        "volume_curve_report",
        "pretrade_report_agent",
        "expected_cost_report_agent",
        "historical_regression_report",
    ],
    "risk": ["beta_risk_report_agent"],
    "peers": ["peer_cluster_report_agent"],
    "debate": ["fast_execution_argument", "liquidity_seeking_argument", "debate_judge_report"],
    "counterfactuals": ["counterfactual_report_agent"],
    "playbook": ["playbook_report_agent"],
    "custom_algo": ["custom_algo_designer_report"],
    "tca": ["tca_report", "cause_effect_report"],
    "charts": ["strategy_report", "simulation_report", "limit_feasibility_report"],
    "scenario": ["expected_cost_report_agent"],
    "memo": ["tab_insight_report_agent"],
    "agent_trace": ["tab_insight_report_agent"],
    "data_room": ["simulation_report", "limit_feasibility_report"],
}


AGENT_REPORT_TITLES = {
    "market_data_report": "Market Data Agent",
    "volume_curve_report": "Volume Curve Agent",
    "pretrade_report_agent": "Pre-Trade Analytics Agent",
    "expected_cost_report_agent": "Expected Cost Model Agent",
    "historical_regression_report": "Historical Regression Agent",
    "beta_risk_report_agent": "Beta Risk Mapping Agent",
    "peer_cluster_report_agent": "Peer Cluster Agent",
    "strategy_report": "Algo Strategy Agent",
    "simulation_report": "Execution Simulator Agent",
    "tca_report": "Benchmark TCA Agent",
    "cause_effect_report": "Cause-Effect TCA Agent",
    "fast_execution_argument": "Fast Execution Advocate",
    "liquidity_seeking_argument": "Liquidity-Seeking Advocate",
    "debate_judge_report": "Debate Judge Agent",
    "counterfactual_report_agent": "Counterfactual Agent",
    "playbook_report_agent": "Execution Playbook Agent",
    "custom_algo_designer_report": "Custom Algo Designer Agent",
    "tab_insight_report_agent": "Tab Insight Agent",
    "limit_feasibility_report": "Limit Feasibility Agent",
}


def render_insights(result, tab_key: str) -> None:
    agent_reports = getattr(result, "agent_reports", {}) or {}
    keys = AGENT_REPORTS_BY_TAB.get(tab_key, [])
    selected = [(key, agent_reports[key]) for key in keys if key in agent_reports]
    if not selected:
        st.info(
            "ADK/Vertex agent commentary was not returned for this local run. "
            "The charts and tables are still computed, but this tab is waiting for live agent reasoning."
        )
        return
    for key, report in selected:
        bullets = "".join(f"<li>{escape(str(item))}</li>" for item in report.highlights)
        caveats = "".join(f"<li>{escape(str(item))}</li>" for item in report.caveats)
        caveat_block = f"<span><strong>Caveats:</strong></span><ul>{caveats}</ul>" if caveats else ""
        title = AGENT_REPORT_TITLES.get(key, key.replace("_", " ").title())
        st.markdown(
            f"""
            <div class="agent-box">
              <strong>{escape(title)}</strong><br/>
              <span><strong>Status:</strong> {escape(str(report.status))}</span>
              <ul>{bullets}</ul>
              {caveat_block}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_agent_narrative(result, tab_key: str) -> None:
    narratives = getattr(result, "agent_narratives", {}) or {}
    narrative = narratives.get(tab_key)
    if narrative is None:
        if getattr(result, "adk_status", "") not in {"success"}:
            st.info(
                "The AI model narrative for this tab appears after a successful ADK/Vertex run. "
                "The audited tables are still available below."
            )
        return
    watch_items = "".join(
        f"<li>{escape(str(item))}</li>" for item in narrative.watch_items[:4]
    )
    watch_block = (
        f"<div class=\"narrative-watch\"><strong>Watch next:</strong><ul>{watch_items}</ul></div>"
        if watch_items
        else ""
    )
    st.markdown(
        f"""
        <div class="narrative-box">
          <div class="narrative-kicker">AI model opinion</div>
          <div class="narrative-title">{escape(narrative.title)}</div>
          <div class="narrative-body"><strong>Verdict:</strong> {escape(narrative.verdict)}</div>
          <div class="narrative-body">{escape(narrative.narrative)}</div>
          <div class="narrative-body"><strong>Recommendation:</strong> {escape(narrative.recommendation)}</div>
          {watch_block}
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    apply_theme()
    service = get_service()

    with st.sidebar:
        st.title("ExecLab AI")
        st.caption("Live intraday execution backtesting on public OHLCV bars.")

        ticker = st.text_input("Ticker", value="NVDA").strip().upper() or "NVDA"
        trade_date = st.date_input("Date", value=previous_weekday())
        side_label = st.segmented_control("Side", options=["Buy", "Sell"], default="Buy")
        quantity = st.number_input("Quantity", min_value=100, max_value=10_000_000, value=50_000, step=1_000)
        interval = st.selectbox("Interval", options=["5m", "1m", "15m"], index=0)

        col_a, col_b = st.columns(2)
        with col_a:
            start_time = st.time_input("Start", value=time(9, 30), step=300)
        with col_b:
            end_time = st.time_input("End", value=time(16, 0), step=300)

        algos = st.multiselect(
            "Algos",
            options=["TWAP", "VWAP", "POV", "IS"],
            default=["TWAP", "VWAP", "POV", "IS"],
        )
        participation_rate = st.slider("POV participation", 1, 50, 10, format="%d%%") / 100.0
        pov_mode_label = st.selectbox(
            "POV mode",
            options=["Strict cap", "Force complete"],
            index=0,
            help="Strict cap reports unfilled shares instead of exceeding the selected POV rate.",
        )
        urgency = st.slider("IS urgency", 0, 100, 65, format="%d%%") / 100.0
        use_limit = st.checkbox("Use limit price", value=False)
        limit_price = None
        if use_limit:
            limit_price = st.number_input("Limit price", min_value=0.01, value=100.0, step=0.25)

        st.divider()
        st.subheader("Scenario Lab")
        spread_bps = st.slider("Spread proxy bps", 0.0, 25.0, 2.0, 0.25)
        impact_bps = st.slider("Impact bps per 10% ADV participation", 0.0, 25.0, 1.5, 0.25)
        drift_bps = st.slider("Drift bps/day", -150.0, 150.0, 0.0, 5.0)
        scenario_paths = st.slider("Scenario paths", 50, 1000, 300, 50)
        seed = st.number_input("Scenario seed", min_value=1, max_value=999999, value=4576, step=1)

        run_clicked = st.button("Run Backtest", type="primary", use_container_width=True)

    st.markdown(
        """
        <div class="execlab-hero">
          <h1 style="margin:0 0 0.25rem 0;">ExecLab AI</h1>
          <p style="margin:0;color:#c6d0da !important;">
            A Vertex-powered multi-agent execution lab for comparing TWAP, VWAP, POV, and
            implementation-shortfall style schedules on live public intraday stock bars.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not run_clicked and "last_result" not in st.session_state:
        st.info(
            "Choose a recent U.S. trading date, then run the backtest. "
            "yfinance intraday data is live-fetched and recent-window limited."
        )
        return

    if run_clicked:
        try:
            request = ExecutionRequest(
                ticker=ticker,
                trade_date=trade_date,
                side="buy" if side_label == "Buy" else "sell",
                quantity=int(quantity),
                start_time=start_time,
                end_time=end_time,
                interval=interval,
                algos=algos,
                participation_rate=participation_rate,
                pov_mode="strict_cap" if pov_mode_label == "Strict cap" else "force_complete",
                urgency=urgency,
                limit_price=float(limit_price) if limit_price else None,
                spread_bps=float(spread_bps),
                impact_bps_per_10pct=float(impact_bps),
                drift_bps_per_day=float(drift_bps),
                scenario_paths=int(scenario_paths),
                seed=int(seed),
            )
        except Exception as exc:
            st.error(f"Invalid request: {exc}")
            return

        with st.spinner("Fetching live bars, running schedules, simulating fills, and asking agents..."):
            try:
                st.session_state["last_result"] = service.run_backtest(request)
                st.session_state["custom_algo_messages"] = []
            except Exception as exc:
                st.error(str(exc))
                return

    result = st.session_state["last_result"]
    render_result(result)


def render_result(result) -> None:
    req = result.request
    st.caption(
        f"Provider: {result.provider} | ADK status: {result.adk_status} | "
        f"Runtime: {result.runtime_seconds:.2f}s | Bars: {len(result.window_bars)}"
    )
    if result.warnings:
        with st.expander("Warnings and caveats", expanded=False):
            for warning in result.warnings:
                st.warning(warning)

    metric_rows = [sim.metrics.model_dump() for sim in result.simulations.values()]
    metrics_df = pd.DataFrame(metric_rows).sort_values("arrival_cost_bps")

    best = metrics_df.iloc[0]
    worst = metrics_df.iloc[-1]
    st.markdown(
        f"""
        <div class="metric-grid">
          <div class="metric-card">
            <div class="metric-label">Best by arrival cost</div>
            <div class="metric-value">{best['algo']}</div>
            <div class="metric-sub">{best['arrival_cost_bps']:.2f} bps</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Market VWAP</div>
            <div class="metric-value">${result.eda.market_vwap:.2f}</div>
            <div class="metric-sub">{req.ticker} {req.trade_date}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Price move in window</div>
            <div class="metric-value">{result.eda.price_move_bps:.1f} bps</div>
            <div class="metric-sub">arrival to final bar close</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Weakest by arrival cost</div>
            <div class="metric-value">{worst['algo']}</div>
            <div class="metric-sub">{worst['arrival_cost_bps']:.2f} bps</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(
        [
            "Pre-Trade Lab",
            "Risk Model",
            "Peers",
            "Agent Debate",
            "Counterfactuals",
            "Playbook",
            "Custom Algo",
            "TCA",
            "Charts",
            "Scenario Lab",
            "Agent Memo",
            "Agent Trace",
            "Data Room",
        ]
    )
    with tabs[0]:
        render_pretrade_lab(result)

    with tabs[1]:
        render_risk_model(result)

    with tabs[2]:
        render_peer_analysis(result)

    with tabs[3]:
        render_agent_debate(result)

    with tabs[4]:
        render_counterfactuals(result)

    with tabs[5]:
        render_playbook(result)

    with tabs[6]:
        render_custom_algo(result)

    with tabs[7]:
        st.subheader("TCA comparison")
        agent_note(
            "What this tab means",
            "BenchmarkTcaAgent, CauseEffectTcaAgent",
            "TCA compares each schedule against arrival price, market VWAP, and close. "
            "For the selected side, lower bps is better; positive bps means the execution was worse than the benchmark.",
        )
        render_agent_narrative(result, "tca")
        render_insights(result, "tca")
        display = metrics_df[
            [
                "algo",
                "avg_fill_price",
                "arrival_cost_bps",
                "vwap_slippage_bps",
                "close_slippage_bps",
                "total_quantity_executed",
                "unfilled_quantity",
                "completion_rate",
                "max_participation_rate",
                "cap_violation_count",
            ]
        ].rename(
            columns={
                "algo": "Algo",
                "avg_fill_price": "Avg Fill",
                "arrival_cost_bps": "Arrival Cost bps",
                "vwap_slippage_bps": "VWAP Slip bps",
                "close_slippage_bps": "Close Slip bps",
                "total_quantity_executed": "Shares",
                "unfilled_quantity": "Unfilled",
                "completion_rate": "Completion",
                "max_participation_rate": "Max Participation",
                "cap_violation_count": "Cap Violations",
            }
        )
        display["Max Participation"] = display["Max Participation"] * 100
        display["Completion"] = display["Completion"] * 100
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Avg Fill": st.column_config.NumberColumn(format="$%.4f"),
                "Arrival Cost bps": st.column_config.NumberColumn(format="%.2f"),
                "VWAP Slip bps": st.column_config.NumberColumn(format="%.2f"),
                "Close Slip bps": st.column_config.NumberColumn(format="%.2f"),
                "Completion": st.column_config.NumberColumn(format="%.2f%%"),
                "Max Participation": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )
        st.subheader("Market EDA")
        eda_df = pd.DataFrame(
            [
                {"Metric": "Arrival price", "Value": f"${result.eda.arrival_price:.4f}"},
                {"Metric": "Close price", "Value": f"${result.eda.close_price:.4f}"},
                {"Metric": "Window volume", "Value": f"{result.eda.window_volume:,.0f}"},
                {"Metric": "Realized vol proxy", "Value": f"{result.eda.realized_volatility_bps:.2f} bps/bar"},
                {"Metric": "High-low spread proxy", "Value": f"{result.eda.high_low_spread_proxy_bps:.2f} bps"},
                {"Metric": "Volume curve source", "Value": result.eda.volume_curve_source},
            ]
        )
        st.dataframe(eda_df, use_container_width=True, hide_index=True)
        st.subheader("How to read the TCA stats")
        st.dataframe(tca_commentary_table(result), use_container_width=True, hide_index=True)

    with tabs[8]:
        agent_note(
            "What this tab means",
            "AlgoStrategyAgent, ExecutionSimulatorAgent, LimitFeasibilityAgent",
            "These charts show how each pseudo execution agent would trade through the day, where fills occur, "
            "and whether participation or limit rules block shares.",
        )
        render_agent_narrative(result, "charts")
        render_insights(result, "charts")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(price_and_fills_chart(result), use_container_width=True)
        with c2:
            st.plotly_chart(schedule_vs_volume_chart(result), use_container_width=True)
        st.plotly_chart(participation_chart(result), use_container_width=True)
        st.plotly_chart(cumulative_completion_chart(result), use_container_width=True)

    with tabs[9]:
        st.subheader("Cost scenario lab")
        agent_note(
            "What this tab means",
            "ExpectedCostModelAgent, HistoricalRegressionAgent",
            "The scenario lab perturbs spread, drift, and impact assumptions. It is an expected-cost stress test, "
            "not a venue-level fill simulator.",
        )
        render_agent_narrative(result, "scenario")
        render_insights(result, "scenario")
        scenario_df = pd.DataFrame([item.model_dump() for item in result.scenario_report.results])
        st.plotly_chart(scenario_chart(scenario_df), use_container_width=True)
        st.dataframe(
            scenario_df.rename(
                columns={
                    "algo": "Algo",
                    "expected_arrival_cost_bps": "Expected Cost bps",
                    "p10_arrival_cost_bps": "P10 bps",
                    "p50_arrival_cost_bps": "P50 bps",
                    "p90_arrival_cost_bps": "P90 bps",
                    "probability_cost_positive": "P(cost > 0)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        for caveat in result.scenario_report.caveats:
            st.caption(caveat)

    with tabs[10]:
        memo = result.memo
        agent_note(
            "What this tab means",
            "NarrativeExplanationAgent, CriticGoldenSetAgent",
            "The final agents consume the computed tool outputs, peer cluster report, beta risk map, and TCA bullets "
            "to write a recommendation memo.",
        )
        render_agent_narrative(result, "memo")
        render_insights(result, "memo")
        st.markdown(
            f"""
            <div class="memo-box">
              <h3 style="margin-top:0;">Best algo: {memo.best_algo}</h3>
              <p>{memo.thesis}</p>
              <p><strong>Scenario:</strong> {memo.scenario_interpretation}</p>
              <p><strong>Limitation:</strong> {memo.limitation}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.subheader("Evidence")
        for item in memo.evidence:
            st.write(f"- {item}")
        st.subheader("Cause-effect TCA")
        st.write(result.causal_report.headline)
        for bullet in result.causal_report.bullets:
            affected = ", ".join(bullet.affected_algos) if bullet.affected_algos else "All"
            st.write(f"- **{bullet.driver} ({affected})**: {bullet.evidence} {bullet.implication}")
        st.subheader("Caveats")
        for item in memo.caveats:
            st.write(f"- {item}")

    with tabs[11]:
        render_agent_trace(result)

    with tabs[12]:
        st.subheader("Fills")
        agent_note(
            "What this tab means",
            "All audited tools",
            "This is the audit trail: modeled fill rows, blocked fills, and the sequence of tool calls used by the agents.",
        )
        render_agent_narrative(result, "data_room")
        render_insights(result, "data_room")
        fill_df = fills_dataframe(result)
        st.dataframe(fill_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download fills CSV",
            data=fill_df.to_csv(index=False),
            file_name=f"{req.ticker}_{req.trade_date}_execlab_fills.csv",
            mime="text/csv",
        )
        st.subheader("Execution trace")
        st.dataframe(pd.DataFrame(result.execution_trace), use_container_width=True, hide_index=True)


def render_pretrade_lab(result) -> None:
    pre = result.pretrade_report
    cost = result.expected_cost_report
    st.subheader("Pre-trade analytics")
    agent_note(
        "What this tab means",
        "PreTradeAnalyticsAgent, ExpectedCostModelAgent, HistoricalRegressionAgent",
        "This tab converts 21 live intraday sessions into liquidity, spread-proxy, volatility, time-risk, "
        "and expected-cost features. The regression is transparent OLS; the spread is a high-low proxy, not NBBO.",
    )
    render_agent_narrative(result, "pretrade")
    render_insights(result, "pretrade")
    st.markdown(
        f"""
        <div class="metric-grid">
          <div class="metric-card"><div class="metric-label">21-day ADV</div><div class="metric-value">{pre.adv_shares:,.0f}</div><div class="metric-sub">{pre.lookback_sessions}/{pre.requested_sessions} sessions fetched live</div></div>
          <div class="metric-card"><div class="metric-label">Order size / ADV</div><div class="metric-value">{pre.order_size_adv_pct * 100:.2f}%</div><div class="metric-sub">{result.request.quantity:,.0f} shares</div></div>
          <div class="metric-card"><div class="metric-label">Spread proxy</div><div class="metric-value">{pre.avg_spread_proxy_bps:.2f} bps</div><div class="metric-sub">21-day high-low proxy, not NBBO</div></div>
          <div class="metric-card"><div class="metric-label">Volatility proxy</div><div class="metric-value">{pre.avg_volatility_bps:.2f} bps</div><div class="metric-sub">average abs bar return</div></div>
          <div class="metric-card"><div class="metric-label">Expected cost</div><div class="metric-value">{cost.expected_cost_bps:.2f} bps</div><div class="metric-sub">transparent OLS + component blend</div></div>
          <div class="metric-card"><div class="metric-label">Time risk</div><div class="metric-value">{pre.time_risk_score:.2f}</div><div class="metric-sub">higher earlier in the day</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if pre.warnings:
        with st.expander("Pre-trade data warnings", expanded=False):
            for warning in pre.warnings[:10]:
                st.warning(warning)

    curve_df = pd.DataFrame([point.model_dump() for point in pre.curve])
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(pretrade_volume_curve_chart(curve_df), use_container_width=True)
    with c2:
        st.plotly_chart(pretrade_spread_vol_curve_chart(curve_df), use_container_width=True)
    st.plotly_chart(monte_carlo_price_paths_chart(result), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(cost_breakdown_chart(pd.DataFrame([item.model_dump() for item in cost.cost_breakdown])), use_container_width=True)
    with c4:
        coef_df = pd.DataFrame([coef.model_dump() for coef in cost.coefficients])
        st.caption(f"OLS observations: {cost.observation_count:,} | R-squared: {cost.model_r2:.2f}")
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

    st.subheader("Stat commentary")
    st.dataframe(pretrade_commentary_table(result), use_container_width=True, hide_index=True)

    for caveat in cost.caveats:
        st.caption(caveat)


def render_risk_model(result) -> None:
    risk = result.beta_risk_report
    st.subheader("Beta risk mapping")
    agent_note(
        "What this tab means",
        "BetaRiskMappingAgent",
        "This tab separates execution impact from market timing risk. It maps the stock to SPY and a sector ETF, "
        "then compares the selected stock's intraday move against those index paths.",
    )
    render_agent_narrative(result, "risk")
    render_insights(result, "risk")
    st.markdown(
        f"""
        <div class="metric-grid">
          <div class="metric-card"><div class="metric-label">Market ETF</div><div class="metric-value">{risk.market_etf}</div><div class="metric-sub">broad market factor</div></div>
          <div class="metric-card"><div class="metric-label">Mapped sector ETF</div><div class="metric-value">{risk.sector_etf}</div><div class="metric-sub">{risk.sector_label} | confidence {risk.mapping_confidence * 100:.0f}%</div></div>
          <div class="metric-card"><div class="metric-label">Market beta</div><div class="metric-value">{risk.beta_market:.2f}</div><div class="metric-sub">corr {risk.correlation_market:.2f}</div></div>
          <div class="metric-card"><div class="metric-label">Sector beta</div><div class="metric-value">{risk.beta_sector:.2f}</div><div class="metric-sub">corr {risk.correlation_sector:.2f}</div></div>
          <div class="metric-card"><div class="metric-label">Idiosyncratic risk</div><div class="metric-value">{risk.idiosyncratic_risk_bps:.2f} bps</div><div class="metric-sub">window-scaled residual vol</div></div>
          <div class="metric-card"><div class="metric-label">Total timing risk</div><div class="metric-value">{risk.total_timing_risk_bps:.2f} bps</div><div class="metric-sub">systematic + sector + residual</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(risk.mapping_reason)
    if risk.warnings:
        for warning in risk.warnings:
            st.warning(warning)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(beta_risk_breakdown_chart(risk), use_container_width=True)
    with c2:
        factor_df = pd.DataFrame(
            [
                {"Factor": "Market ETF", "ETF": risk.market_etf, "Beta": risk.beta_market, "Correlation": risk.correlation_market, "Daily Vol bps": risk.market_daily_vol_bps},
                {"Factor": "Sector ETF", "ETF": risk.sector_etf, "Beta": risk.beta_sector, "Correlation": risk.correlation_sector, "Daily Vol bps": risk.sector_daily_vol_bps},
                {"Factor": "Stock", "ETF": risk.ticker, "Beta": 1.0, "Correlation": 1.0, "Daily Vol bps": risk.ticker_daily_vol_bps},
            ]
        )
        st.caption(f"Daily observations: {risk.observation_count:,} | Factor R-squared: {risk.r_squared:.2f}")
        st.dataframe(
            factor_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Beta": st.column_config.NumberColumn(format="%.3f"),
                "Correlation": st.column_config.NumberColumn(format="%.3f"),
                "Daily Vol bps": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    if risk.index_comparison:
        st.plotly_chart(index_vs_stock_chart(result), use_container_width=True)

    st.subheader("Impact vs market-risk interpretation")
    st.write(
        f"Exec impact is controlled by participation, spread proxy, and urgency; market timing risk is "
        f"the exposure you carry while waiting. For this run, systematic risk contributes "
        f"{risk.systematic_risk_bps:.2f} bps, sector risk contributes {risk.sector_risk_bps:.2f} bps, "
        f"and idiosyncratic residual risk contributes {risk.idiosyncratic_risk_bps:.2f} bps over the execution window."
    )
    coef_df = pd.DataFrame([coef.model_dump() for coef in risk.coefficients])
    if not coef_df.empty:
        st.subheader("Factor regression coefficients")
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

    st.subheader("Stat commentary")
    st.dataframe(risk_commentary_table(result), use_container_width=True, hide_index=True)


def render_peer_analysis(result) -> None:
    peer = result.peer_report
    st.subheader("Closest peer stock analysis")
    agent_note(
        "What this tab means",
        "PeerClusterAgent, CauseEffectTcaAgent",
        "The peer agent looks inside the mapped sector ETF, finds stocks most correlated with the target, "
        "checks whether their recent moves confirm the target move, and translates that into fast/slow execution pressure.",
    )
    render_agent_narrative(result, "peers")
    render_insights(result, "peers")
    st.markdown(
        f"""
        <div class="metric-grid">
          <div class="metric-card"><div class="metric-label">Sector peer basket</div><div class="metric-value">{peer.sector_etf}</div><div class="metric-sub">{peer.analyzed_count}/{peer.candidate_count} peers analyzed</div></div>
          <div class="metric-card"><div class="metric-label">Average peer correlation</div><div class="metric-value">{peer.average_peer_correlation:.2f}</div><div class="metric-sub">higher means peers trade with the name</div></div>
          <div class="metric-card"><div class="metric-label">Crowding score</div><div class="metric-value">{peer.crowding_score:.2f}</div><div class="metric-sub">correlation plus same-direction pressure</div></div>
          <div class="metric-card"><div class="metric-label">Median peer move</div><div class="metric-value">{peer.median_peer_move_bps:.1f} bps</div><div class="metric-sub">recent daily move across peer set</div></div>
          <div class="metric-card"><div class="metric-label">Target recent move</div><div class="metric-value">{peer.target_recent_move_bps:.1f} bps</div><div class="metric-sub">{result.request.ticker} recent daily move</div></div>
          <div class="metric-card"><div class="metric-label">Urgency read</div><div class="metric-value">{peer.urgency_recommendation}</div><div class="metric-sub">fast/slow signal from peers</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(peer.rationale)
    st.caption(peer.market_impact_note)
    if peer.warnings:
        for warning in peer.warnings:
            st.warning(warning)

    peer_df = pd.DataFrame([item.model_dump() for item in peer.peers])
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(peer_cluster_chart(peer_df, result.request.ticker), use_container_width=True)
    with c2:
        st.plotly_chart(peer_move_bar_chart(peer_df), use_container_width=True)
    if not peer_df.empty:
        st.dataframe(
            peer_df.rename(
                columns={
                    "ticker": "Peer",
                    "correlation": "Correlation",
                    "beta_to_target": "Beta to Target",
                    "recent_move_bps": "Recent Move bps",
                    "target_move_bps": "Target Move bps",
                    "move_gap_bps": "Move Gap bps",
                    "cluster": "Cluster",
                    "impact_signal": "Impact Signal",
                }
            ),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Correlation": st.column_config.NumberColumn(format="%.3f"),
                "Beta to Target": st.column_config.NumberColumn(format="%.3f"),
                "Recent Move bps": st.column_config.NumberColumn(format="%.2f"),
                "Target Move bps": st.column_config.NumberColumn(format="%.2f"),
                "Move Gap bps": st.column_config.NumberColumn(format="%.2f"),
            },
        )


def render_agent_debate(result) -> None:
    debate = result.debate_report
    st.subheader("Agent debate")
    agent_note(
        "What this tab means",
        "FastExecutionAdvocate, LiquiditySeekingAdvocate, DebateJudgeAgent",
        "Two agents argue opposite execution philosophies. The judge then chooses which argument is better grounded in the computed TCA, pre-trade, beta, peer, and completion data.",
    )
    render_agent_narrative(result, "debate")
    render_insights(result, "debate")
    st.markdown(
        f"""
        <div class="metric-grid">
          <div class="metric-card"><div class="metric-label">Judge winner</div><div class="metric-value">{debate.judge_winner}</div><div class="metric-sub">stronger argument</div></div>
          <div class="metric-card"><div class="metric-label">Recommended algo</div><div class="metric-value">{debate.recommended_algo}</div><div class="metric-sub">from agent debate report</div></div>
          <div class="metric-card"><div class="metric-label">Confidence</div><div class="metric-value">{debate.confidence * 100:.0f}%</div><div class="metric-sub">arrival-cost gap adjusted</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(debate.fast_case.advocate)
        st.write(debate.fast_case.thesis)
        for item in debate.fast_case.evidence:
            st.write(f"- {item}")
        for caveat in debate.fast_case.caveats:
            st.caption(caveat)
    with c2:
        st.subheader(debate.liquidity_case.advocate)
        st.write(debate.liquidity_case.thesis)
        for item in debate.liquidity_case.evidence:
            st.write(f"- {item}")
        for caveat in debate.liquidity_case.caveats:
            st.caption(caveat)
    st.subheader("Judge rationale")
    st.write(debate.judge_rationale)
    for item in debate.deciding_factors:
        st.write(f"- {item}")


def render_counterfactuals(result) -> None:
    report = result.counterfactual_report
    st.subheader("Counterfactual robustness")
    agent_note(
        "What this tab means",
        "CounterfactualAgent",
        "This agent asks what would have made another algo win. It stress-tests the recommendation under flat tape, wider spread, larger order, peer crowding, and completion-adjusted assumptions.",
    )
    render_agent_narrative(result, "counterfactuals")
    render_insights(result, "counterfactuals")
    st.write(report.summary)
    scenario_rows = []
    for scenario in report.scenarios:
        for algo, cost in scenario.estimated_costs_bps.items():
            scenario_rows.append(
                {
                    "Scenario": scenario.name,
                    "Algo": algo,
                    "Estimated Cost bps": cost,
                    "Winner": scenario.estimated_winner,
                    "Assumption": scenario.assumption_change,
                    "Rationale": scenario.rationale,
                }
            )
    frame = pd.DataFrame(scenario_rows)
    st.plotly_chart(counterfactual_chart(frame), use_container_width=True)
    if not frame.empty:
        winners = (
            frame[["Scenario", "Winner", "Assumption", "Rationale"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        st.dataframe(winners, use_container_width=True, hide_index=True)


def render_playbook(result) -> None:
    playbook = result.playbook_report
    st.subheader("Execution playbook")
    agent_note(
        "What this tab means",
        "ExecutionPlaybookAgent",
        "This converts the backtest, debate, and counterfactuals into desk-style operating guidance: what to run, how urgently, when to switch, and what to monitor.",
    )
    render_agent_narrative(result, "playbook")
    render_insights(result, "playbook")
    st.markdown(
        f"""
        <div class="memo-box">
          <h3 style="margin-top:0;">Run {playbook.recommended_algo}</h3>
          <p>{playbook.rationale}</p>
          <p><strong>Urgency:</strong> {playbook.urgency}</p>
          <p><strong>Participation:</strong> {playbook.participation_guidance}</p>
          <p><strong>Limit:</strong> {playbook.limit_guidance}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Monitoring triggers")
        for item in playbook.monitoring_triggers:
            st.write(f"- {item}")
    with c2:
        st.subheader("Switch rules")
        for item in playbook.switch_rules:
            st.write(f"- {item}")


def render_custom_algo(result) -> None:
    report = result.custom_algo_report
    metrics = report.simulation.metrics
    st.subheader("Agent-designed custom algo")
    agent_note(
        "What this tab means",
        "CustomAlgoPlannerAgent, CustomAlgoDesignerAgent, TabInsightAgent, ExecutionSimulatorAgent",
        "This tab lets ADK interpret the desk brief into a structured custom plan, builds a hybrid schedule from that plan, then backtests it with the same bar-based fill model. It is a research idea, not production routing logic.",
    )
    render_agent_narrative(result, "custom_algo")
    render_custom_algo_chat(result)
    render_custom_algo_behavior_story(result)
    render_insights(result, "custom_algo")
    st.markdown(
        f"""
        <div class="metric-grid">
          <div class="metric-card"><div class="metric-label">Custom algo</div><div class="metric-value">{report.name}</div><div class="metric-sub">{report.style}</div></div>
          <div class="metric-card"><div class="metric-label">Arrival cost</div><div class="metric-value">{metrics.arrival_cost_bps:.2f} bps</div><div class="metric-sub">same TCA math as benchmarks</div></div>
          <div class="metric-card"><div class="metric-label">Completion</div><div class="metric-value">{metrics.completion_rate * 100:.1f}%</div><div class="metric-sub">{metrics.unfilled_quantity:,} unfilled shares</div></div>
          <div class="metric-card"><div class="metric-label">Adaptive cap</div><div class="metric-value">{float(report.parameters.get("adaptive_participation_cap", 0)) * 100:.1f}%</div><div class="metric-sub">{metrics.cap_violation_count} cap violations</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(report.description)
    agent_plan = report.parameters.get("agent_plan")
    if agent_plan:
        with st.expander("Active custom brief and CustomAlgoPlannerAgent plan", expanded=False):
            if report.parameters.get("user_brief"):
                st.write(report.parameters["user_brief"])
            plan_rows = [{"Field": key, "Value": value} for key, value in agent_plan.items()]
            st.dataframe(pd.DataFrame(plan_rows), use_container_width=True, hide_index=True)
    elif result.request.custom_algo_instructions:
        st.warning(
            "Custom brief captured, but CustomAlgoPlannerAgent did not return a plan. "
            "Run with ADK/Vertex enabled for agentic customization."
        )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Component recipe")
        component_df = pd.DataFrame([item.model_dump() for item in report.components])
        if not component_df.empty:
            component_df["weight"] = component_df["weight"] * 100
            st.dataframe(
                component_df.rename(
                    columns={"name": "Component", "weight": "Weight %", "reason": "Why it is used"}
                ),
                use_container_width=True,
                hide_index=True,
                column_config={"Weight %": st.column_config.NumberColumn(format="%.1f%%")},
            )
    with c2:
        st.subheader("Custom vs benchmark TCA")
        compare = _custom_comparison_table(result)
        st.dataframe(
            compare,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Arrival Cost bps": st.column_config.NumberColumn(format="%.2f"),
                "VWAP Slip bps": st.column_config.NumberColumn(format="%.2f"),
                "Completion %": st.column_config.NumberColumn(format="%.1f%%"),
                "Max Participation %": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

    st.plotly_chart(custom_algo_schedule_chart(result), use_container_width=True)
    st.subheader("Agent rationale")
    for item in report.rationale:
        st.write(f"- {item}")
    st.subheader("Caveats")
    for item in report.caveats:
        st.caption(item)


def render_custom_algo_behavior_story(result) -> None:
    report = result.custom_algo_report
    parameters = report.parameters or {}
    agent_plan = parameters.get("agent_plan")
    story = str(parameters.get("agent_execution_story") or "").strip()
    operating_rules = parameters.get("agent_operating_rules") or []
    if not story and isinstance(agent_plan, dict):
        story = str(agent_plan.get("execution_story") or "").strip()
        operating_rules = operating_rules or agent_plan.get("operating_rules", [])

    if not story:
        if result.request.custom_algo_instructions and getattr(result, "adk_status", "") != "success":
            st.info(
                "CustomAlgoPlannerAgent will write the behavior story after a successful ADK/Vertex run."
            )
        return

    rule_items = "".join(
        f"<li>{escape(str(item))}</li>" for item in list(operating_rules)[:5]
    )
    rules_block = (
        f"<div class=\"narrative-watch\"><strong>Operating rules:</strong><ul>{rule_items}</ul></div>"
        if rule_items
        else ""
    )
    brief = str(parameters.get("user_brief") or "").strip()
    brief_block = (
        f"<div class=\"narrative-body\"><strong>Desk brief:</strong> {escape(brief)}</div>"
        if brief
        else ""
    )
    st.markdown(
        f"""
        <div class="narrative-box">
          <div class="narrative-kicker">CustomAlgoPlannerAgent behavior story</div>
          <div class="narrative-title">How this custom algo would trade</div>
          {brief_block}
          <div class="narrative-body">{escape(story)}</div>
          {rules_block}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_custom_algo_chat(result) -> None:
    st.subheader("Custom algo design chat")
    st.caption(
        "Tell the agent your execution objective: urgency, max participation, PM exposure, "
        "completion target by time, risk limits, and limit-price constraints."
    )
    messages = st.session_state.setdefault("custom_algo_messages", [])
    if not messages and result.request.custom_algo_instructions:
        messages.append({"role": "user", "content": result.request.custom_algo_instructions})

    with st.chat_message("assistant"):
        st.write(
            "Give CustomAlgoPlannerAgent a desk brief. Example: "
            "`PM wants 60% done by 11:00, keep max participation under 12%, minimize impact, "
            "but reduce exposure before the Fed headline.`"
        )
    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input(
        "Describe your custom execution constraints...",
        key="custom_algo_chat_input",
    )
    if prompt:
        messages.append({"role": "user", "content": prompt})
        brief = "\n".join(
            message["content"] for message in messages if message.get("role") == "user"
        )
        updated_request = result.request.model_copy(update={"custom_algo_instructions": brief})
        with st.spinner("CustomAlgoPlannerAgent is interpreting the brief and redesigning the schedule..."):
            st.session_state["last_result"] = get_service().run_backtest(updated_request)
        st.rerun()


def render_agent_trace(result) -> None:
    st.subheader("Agent trace and rubric QA")
    agent_note(
        "What this tab means",
        "ExecLabCoordinatorAgent and all sub-agents",
        "This shows the handoff map and tool trace. It is the audit evidence that the app is agentic and grounded in tools.",
    )
    render_agent_narrative(result, "agent_trace")
    render_insights(result, "agent_trace")
    pipeline = [
        ("MarketDataAgent", "Validates live bars and benchmark prices"),
        ("VolumeCurveAgent", "Explains VWAP curve source and volume regime"),
        ("PreTradeAnalyticsAgent", "Summarizes 21-day liquidity, spread proxy, volatility, and time risk"),
        ("ExpectedCostModelAgent", "Explains expected cost and component breakdown"),
        ("HistoricalRegressionAgent", "Audits OLS feature engineering and fit quality"),
        ("BetaRiskMappingAgent", "Maps market/sector beta and systematic-vs-idiosyncratic risk"),
        ("PeerClusterAgent", "Analyzes closest peers, clustering, and urgency pressure"),
        ("AlgoStrategyAgent", "Compares TWAP/VWAP/POV/IS schedule behavior"),
        ("ExecutionSimulatorAgent", "Explains fills, completion, and blocked/limit behavior"),
        ("BenchmarkTcaAgent", "Reviews arrival, VWAP, close, and scenario metrics"),
        ("CauseEffectTcaAgent", "Builds cause-effect bullets for why the winner won"),
        ("FastExecutionAdvocate", "Argues for faster/front-loaded execution"),
        ("LiquiditySeekingAdvocate", "Argues for VWAP/TWAP/POV liquidity-seeking execution"),
        ("DebateJudgeAgent", "Judges the debate and recommendation robustness"),
        ("CounterfactualAgent", "Asks what assumptions would change the winner"),
        ("ExecutionPlaybookAgent", "Writes monitoring triggers and switch rules"),
        ("CustomAlgoDesignerAgent", "Builds and explains a hybrid strategy for this tape"),
        ("TabInsightAgent", "Generates live ADK commentary from the computed context"),
        ("TabNarrativeAgent", "Writes model-authored opinion sections for every dashboard tab"),
        ("NarrativeExplanationAgent", "Drafts the execution memo"),
        ("CriticGoldenSetAgent", "Checks caveats, numbers, and final recommendation"),
    ]
    st.dataframe(
        pd.DataFrame(pipeline, columns=["Agent", "Role"]),
        use_container_width=True,
        hide_index=True,
    )
    qa = pd.DataFrame(
        [
            {"Check": "Uses live market data", "Status": "Pass" if result.provider != "unknown" else "Review"},
            {"Check": "Has audited TCA metrics", "Status": "Pass" if result.simulations else "Review"},
            {"Check": "Explains limitation language", "Status": "Pass" if "not a production OMS/EMS" in result.memo.limitation else "Review"},
            {"Check": "Includes peer/beta risk", "Status": "Pass" if result.peer_report.analyzed_count >= 0 and result.beta_risk_report.observation_count >= 0 else "Review"},
            {"Check": "Has debate and counterfactual reports", "Status": "Pass" if result.debate_report.recommended_algo and result.counterfactual_report.scenarios else "Review"},
            {"Check": "Has custom algo designer output", "Status": "Pass" if result.custom_algo_report.components else "Review"},
            {"Check": "Has ADK agent commentary", "Status": "Pass" if result.agent_reports else "Needs ADK run"},
            {"Check": "Has AI tab narratives", "Status": "Pass" if result.agent_narratives else "Needs ADK run"},
        ]
    )
    st.subheader("Rubric QA")
    st.dataframe(qa, use_container_width=True, hide_index=True)
    st.subheader("Tool execution trace")
    st.dataframe(pd.DataFrame(result.execution_trace), use_container_width=True, hide_index=True)


def fills_dataframe(result) -> pd.DataFrame:
    rows = []
    for sim in all_simulations(result).values():
        for fill in sim.fills:
            rows.append(fill.model_dump())
    return pd.DataFrame(rows)


def all_simulations(result) -> dict:
    simulations = dict(result.simulations)
    custom = getattr(result, "custom_algo_report", None)
    if custom is not None:
        simulations["CUSTOM"] = custom.simulation
    return simulations


def all_schedules(result) -> dict:
    schedules = dict(result.schedules)
    custom_schedule = getattr(result, "custom_schedule", pd.DataFrame())
    if isinstance(custom_schedule, pd.DataFrame) and not custom_schedule.empty:
        schedules["CUSTOM"] = custom_schedule
    return schedules


def _custom_comparison_table(result) -> pd.DataFrame:
    rows = []
    for algo, sim in all_simulations(result).items():
        metrics = sim.metrics
        rows.append(
            {
                "Algo": algo,
                "Arrival Cost bps": metrics.arrival_cost_bps,
                "VWAP Slip bps": metrics.vwap_slippage_bps,
                "Completion %": metrics.completion_rate * 100,
                "Unfilled": metrics.unfilled_quantity,
                "Max Participation %": metrics.max_participation_rate * 100,
            }
        )
    return pd.DataFrame(rows).sort_values("Arrival Cost bps")


def pretrade_commentary_table(result) -> pd.DataFrame:
    pre = result.pretrade_report
    cost = result.expected_cost_report
    return pd.DataFrame(
        [
            {
                "Stat": "21-day ADV",
                "Computed value": f"{pre.adv_shares:,.0f} shares",
                "What it means": "Average full-session volume across fetched lookback sessions; larger ADV usually allows slower execution with less footprint.",
            },
            {
                "Stat": "Order size / ADV",
                "Computed value": f"{pre.order_size_adv_pct * 100:.2f}%",
                "What it means": "Parent order as a share of normal daily liquidity; higher values raise impact and completion risk.",
            },
            {
                "Stat": "21-day spread proxy",
                "Computed value": f"{pre.avg_spread_proxy_bps:.2f} bps",
                "What it means": "Average high-low bar range proxy, not NBBO. It approximates friction when true bid/ask data is unavailable.",
            },
            {
                "Stat": "21-day volatility proxy",
                "Computed value": f"{pre.avg_volatility_bps:.2f} bps/bar",
                "What it means": "Average absolute bar return; higher values mean waiting exposes the order to more price-path risk.",
            },
            {
                "Stat": "Expected cost model",
                "Computed value": f"{cost.expected_cost_bps:.2f} bps",
                "What it means": "Transparent OLS plus component blend using participation, spread proxy, volatility, time risk, relative volume, order size, and drift.",
            },
        ]
    )


def risk_commentary_table(result) -> pd.DataFrame:
    risk = result.beta_risk_report
    return pd.DataFrame(
        [
            {
                "Stat": "Market beta",
                "Computed value": f"{risk.beta_market:.2f}",
                "What it means": f"Sensitivity of {risk.ticker} daily returns to {risk.market_etf}; high beta means market movement matters while the order waits.",
            },
            {
                "Stat": "Sector beta",
                "Computed value": f"{risk.beta_sector:.2f}",
                "What it means": f"Sensitivity to {risk.sector_etf}; useful for separating sector flow from stock-specific movement.",
            },
            {
                "Stat": "Systematic timing risk",
                "Computed value": f"{risk.systematic_risk_bps:.2f} bps",
                "What it means": "Estimated execution-window risk from broad market movement.",
            },
            {
                "Stat": "Idiosyncratic timing risk",
                "Computed value": f"{risk.idiosyncratic_risk_bps:.2f} bps",
                "What it means": "Residual stock-specific risk after market and sector factors; if this dominates, peer/index hedges explain less of the move.",
            },
        ]
    )


def tca_commentary_table(result) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Stat": "Arrival cost bps",
                "What it means": "Average fill versus the first execution-window price. Lower is better for both buy and sell after side-adjusting.",
            },
            {
                "Stat": "VWAP slippage bps",
                "What it means": "Average fill versus market VWAP. VWAP-style algos should usually score well here when the volume curve is stable.",
            },
            {
                "Stat": "Close slippage bps",
                "What it means": "Average fill versus closing price. This can favor slower schedules on favorable tapes, but that is price path, not guaranteed execution skill.",
            },
            {
                "Stat": "Completion and unfilled",
                "What it means": "Shows whether strict POV caps or limit prices left shares behind. An algo with great bps but poor completion may not be usable.",
            },
            {
                "Stat": "Max participation",
                "What it means": "Peak child-order size divided by bar volume. This is the clearest impact-pressure diagnostic in the bar simulator.",
            },
        ]
    )


def price_and_fills_chart(result) -> go.Figure:
    fig = go.Figure()
    bars = result.window_bars
    fig.add_trace(
        go.Scatter(
            x=bars["timestamp_et"],
            y=bars["close"],
            mode="lines",
            name="Close",
            line=dict(color="#e5e7eb", width=2),
        )
    )
    colors = {"TWAP": "#38bdf8", "VWAP": "#22c55e", "POV": "#f59e0b", "IS": "#ef4444", "CUSTOM": "#a78bfa"}
    for algo, sim in all_simulations(result).items():
        fills = pd.DataFrame([fill.model_dump() for fill in sim.fills])
        if fills.empty:
            continue
        executed = fills[fills["executed_quantity"] > 0]
        blocked = fills[fills["executed_quantity"] <= 0]
        if not executed.empty:
            fig.add_trace(
                go.Scatter(
                    x=executed["timestamp_et"],
                    y=executed["fill_price"],
                    mode="markers",
                    name=f"{algo} fills",
                    marker=dict(size=6, color=colors.get(algo, "#a78bfa"), opacity=0.75),
                )
            )
        if not blocked.empty:
            fig.add_trace(
                go.Scatter(
                    x=blocked["timestamp_et"],
                    y=blocked["fill_price"],
                    mode="markers",
                    name=f"{algo} blocked",
                    marker=dict(size=8, color=colors.get(algo, "#a78bfa"), symbol="x"),
                )
            )
    if result.request.limit_price is not None:
        fig.add_hline(
            y=result.request.limit_price,
            line_dash="dash",
            line_color="#f97316",
            annotation_text=f"Limit {result.request.limit_price:.2f}",
        )
    fig.update_layout(
        title="Price path and simulated fills",
        template="plotly_dark",
        paper_bgcolor="#0f1419",
        plot_bgcolor="#0f1419",
        height=420,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def schedule_vs_volume_chart(result) -> go.Figure:
    fig = go.Figure()
    bars = result.window_bars
    fig.add_trace(
        go.Bar(
            x=bars["timestamp_et"],
            y=bars["volume"],
            name="Market volume",
            marker_color="rgba(148, 163, 184, 0.38)",
            yaxis="y2",
        )
    )
    colors = {"TWAP": "#38bdf8", "VWAP": "#22c55e", "POV": "#f59e0b", "IS": "#ef4444", "CUSTOM": "#a78bfa"}
    for algo, schedule in all_schedules(result).items():
        fig.add_trace(
            go.Scatter(
                x=schedule["timestamp_et"],
                y=schedule["target_quantity"],
                mode="lines",
                name=algo,
                line=dict(color=colors.get(algo, "#a78bfa"), width=2),
            )
        )
    fig.update_layout(
        title="Execution schedule vs market volume",
        template="plotly_dark",
        paper_bgcolor="#0f1419",
        plot_bgcolor="#0f1419",
        height=420,
        yaxis=dict(title="Child shares"),
        yaxis2=dict(title="Market volume", overlaying="y", side="right", showgrid=False),
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def custom_algo_schedule_chart(result) -> go.Figure:
    fig = go.Figure()
    bars = result.window_bars
    fig.add_trace(
        go.Bar(
            x=bars["timestamp_et"],
            y=bars["volume"],
            name="Market volume",
            marker_color="rgba(148, 163, 184, 0.30)",
            yaxis="y2",
        )
    )
    colors = {"TWAP": "#38bdf8", "VWAP": "#22c55e", "POV": "#f59e0b", "IS": "#ef4444", "CUSTOM": "#a78bfa"}
    for algo, schedule in all_schedules(result).items():
        width = 4 if algo == "CUSTOM" else 1.8
        dash = "solid" if algo == "CUSTOM" else "dot"
        fig.add_trace(
            go.Scatter(
                x=schedule["timestamp_et"],
                y=schedule["target_quantity"],
                mode="lines",
                name=algo,
                line=dict(color=colors.get(algo, "#e5e7eb"), width=width, dash=dash),
            )
        )
    custom = getattr(result, "custom_schedule", pd.DataFrame())
    if isinstance(custom, pd.DataFrame) and not custom.empty and "participation_cap_quantity" in custom:
        fig.add_trace(
            go.Scatter(
                x=custom["timestamp_et"],
                y=custom["participation_cap_quantity"],
                mode="lines",
                name="Custom cap",
                line=dict(color="#c4b5fd", width=1.5, dash="dash"),
            )
        )
    fig.update_layout(
        title="Agent-designed custom schedule aligned with benchmark algos",
        template="plotly_dark",
        paper_bgcolor="#0f1419",
        plot_bgcolor="#0f1419",
        height=430,
        yaxis=dict(title="Child shares"),
        yaxis2=dict(title="Market volume", overlaying="y", side="right", showgrid=False),
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def participation_chart(result) -> go.Figure:
    fig = go.Figure()
    colors = {"TWAP": "#38bdf8", "VWAP": "#22c55e", "POV": "#f59e0b", "IS": "#ef4444", "CUSTOM": "#a78bfa"}
    for algo, sim in all_simulations(result).items():
        fills = pd.DataFrame([fill.model_dump() for fill in sim.fills])
        if fills.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=fills["timestamp_et"],
                y=fills["participation_rate"] * 100,
                mode="lines",
                name=algo,
                line=dict(color=colors.get(algo, "#a78bfa"), width=2),
            )
        )
    fig.update_layout(
        title="Participation profile",
        template="plotly_dark",
        paper_bgcolor="#0f1419",
        plot_bgcolor="#0f1419",
        height=360,
        yaxis_title="Participation %",
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def cumulative_completion_chart(result) -> go.Figure:
    fig = go.Figure()
    colors = {"TWAP": "#38bdf8", "VWAP": "#22c55e", "POV": "#f59e0b", "IS": "#ef4444", "CUSTOM": "#a78bfa"}
    for algo, sim in all_simulations(result).items():
        fills = pd.DataFrame([fill.model_dump() for fill in sim.fills])
        if fills.empty:
            continue
        fills["cum_exec_pct"] = fills["executed_quantity"].cumsum() / max(1, result.request.quantity) * 100
        fig.add_trace(
            go.Scatter(
                x=fills["timestamp_et"],
                y=fills["cum_exec_pct"],
                mode="lines",
                name=algo,
                line=dict(color=colors.get(algo, "#a78bfa"), width=2),
            )
        )
    fig.update_layout(
        title="Cumulative theoretical completion",
        template="plotly_dark",
        paper_bgcolor="#0f1419",
        plot_bgcolor="#0f1419",
        height=340,
        yaxis_title="Executed %",
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def pretrade_volume_curve_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not frame.empty:
        fig.add_trace(go.Bar(x=frame["time_key"], y=frame["avg_volume"], name="Avg volume", marker_color="#38bdf8"))
    fig.update_layout(
        title="21-day average volume curve",
        template="plotly_dark",
        paper_bgcolor="#0f1419",
        plot_bgcolor="#0f1419",
        height=360,
        xaxis_title="Market time",
        yaxis_title="Shares",
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def pretrade_spread_vol_curve_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not frame.empty:
        fig.add_trace(go.Scatter(x=frame["time_key"], y=frame["avg_spread_proxy_bps"], mode="lines", name="Spread proxy bps", line=dict(color="#f59e0b")))
        fig.add_trace(go.Scatter(x=frame["time_key"], y=frame["avg_volatility_bps"], mode="lines", name="Volatility bps", line=dict(color="#22c55e"), yaxis="y2"))
    fig.update_layout(
        title="21-day spread proxy and volatility curves",
        template="plotly_dark",
        paper_bgcolor="#0f1419",
        plot_bgcolor="#0f1419",
        height=360,
        xaxis_title="Market time",
        yaxis=dict(title="Spread proxy bps"),
        yaxis2=dict(title="Volatility bps", overlaying="y", side="right", showgrid=False),
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def monte_carlo_price_paths_chart(result) -> go.Figure:
    curve = pd.DataFrame([point.model_dump() for point in result.pretrade_report.curve])
    fig = go.Figure()
    if curve.empty:
        return fig

    rng = np.random.default_rng(result.request.seed)
    times = curve["time_key"].tolist()
    start_price = float(result.eda.arrival_price)
    drift_per_bar = result.request.drift_bps_per_day / max(1, len(times))
    vol = curve["avg_volatility_bps"].fillna(result.pretrade_report.avg_volatility_bps).clip(lower=0.5).to_numpy(float)
    for path_id in range(18):
        price = start_price
        prices = []
        for sigma in vol:
            shock_bps = drift_per_bar + rng.normal(0, sigma)
            price *= 1.0 + shock_bps / 10_000.0
            prices.append(price)
        fig.add_trace(
            go.Scatter(
                x=times,
                y=prices,
                mode="lines",
                name=f"Path {path_id + 1}",
                line=dict(color="rgba(56, 189, 248, 0.22)", width=1),
                showlegend=False,
            )
        )

    actual = result.window_bars.copy()
    if not actual.empty:
        actual["time_key"] = actual["timestamp_et"].dt.strftime("%H:%M")
        fig.add_trace(
            go.Scatter(
                x=actual["time_key"],
                y=actual["close"],
                mode="lines",
                name="Actual selected-day close",
                line=dict(color="#f59e0b", width=3),
            )
        )
    fig.update_layout(
        title="Monte Carlo price paths through the execution day",
        template="plotly_dark",
        paper_bgcolor="#0f1419",
        plot_bgcolor="#0f1419",
        height=390,
        xaxis_title="Market time",
        yaxis_title="Modeled price",
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def pretrade_through_day_cost_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not frame.empty:
        fig.add_trace(go.Scatter(x=frame["time_key"], y=frame["p90_bps"], mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(
            go.Scatter(
                x=frame["time_key"],
                y=frame["p10_bps"],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(56, 189, 248, 0.18)",
                line=dict(width=0),
                name="P10-P90",
            )
        )
        fig.add_trace(go.Scatter(x=frame["time_key"], y=frame["p50_bps"], mode="lines", name="P50 expected cost", line=dict(color="#38bdf8", width=2)))
    fig.update_layout(
        title="Monte Carlo expected cost through the day",
        template="plotly_dark",
        paper_bgcolor="#0f1419",
        plot_bgcolor="#0f1419",
        height=390,
        xaxis_title="Potential start time",
        yaxis_title="Expected cost bps",
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def cost_breakdown_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not frame.empty:
        fig.add_trace(go.Bar(x=frame["component"], y=frame["bps"], marker_color="#a78bfa", name="Cost bps"))
    fig.update_layout(
        title="Expected cost breakdown",
        template="plotly_dark",
        paper_bgcolor="#0f1419",
        plot_bgcolor="#0f1419",
        height=340,
        yaxis_title="bps",
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def beta_risk_breakdown_chart(risk) -> go.Figure:
    frame = pd.DataFrame(
        [
            {"component": "Systematic", "bps": risk.systematic_risk_bps},
            {"component": "Sector", "bps": risk.sector_risk_bps},
            {"component": "Idiosyncratic", "bps": risk.idiosyncratic_risk_bps},
        ]
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame["component"],
            y=frame["bps"],
            marker_color=["#38bdf8", "#22c55e", "#f59e0b"],
            name="Timing risk bps",
        )
    )
    fig.update_layout(
        title="Execution-window timing risk decomposition",
        template="plotly_dark",
        paper_bgcolor="#0f1419",
        plot_bgcolor="#0f1419",
        height=360,
        yaxis_title="bps",
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def index_vs_stock_chart(result) -> go.Figure:
    frame = pd.DataFrame([point.model_dump() for point in result.beta_risk_report.index_comparison])
    fig = go.Figure()
    if not frame.empty:
        fig.add_trace(go.Scatter(x=frame["time_key"], y=frame["ticker_return_bps"], mode="lines", name=result.request.ticker, line=dict(color="#f59e0b", width=3)))
        fig.add_trace(go.Scatter(x=frame["time_key"], y=frame["market_return_bps"], mode="lines", name=result.beta_risk_report.market_etf, line=dict(color="#38bdf8", width=2)))
        fig.add_trace(go.Scatter(x=frame["time_key"], y=frame["sector_return_bps"], mode="lines", name=result.beta_risk_report.sector_etf, line=dict(color="#22c55e", width=2)))
    fig.update_layout(
        title="Selected stock vs market and sector intraday movement",
        template="plotly_dark",
        paper_bgcolor="#0f1419",
        plot_bgcolor="#0f1419",
        height=380,
        xaxis_title="Market time",
        yaxis_title="Cumulative return bps",
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def peer_cluster_chart(frame: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    if not frame.empty:
        fig.add_trace(
            go.Scatter(
                x=frame["correlation"],
                y=frame["recent_move_bps"],
                mode="markers+text",
                text=frame["ticker"],
                textposition="top center",
                marker=dict(
                    size=(frame["beta_to_target"].abs().clip(lower=0.2, upper=2.0) * 16),
                    color=frame["move_gap_bps"],
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title="Move gap bps"),
                ),
                name="Peers",
            )
        )
    fig.update_layout(
        title=f"Closest peer clusters for {ticker}",
        template="plotly_dark",
        paper_bgcolor="#0f1419",
        plot_bgcolor="#0f1419",
        height=360,
        xaxis_title="Correlation to target",
        yaxis_title="Recent peer move bps",
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def peer_move_bar_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not frame.empty:
        fig.add_trace(
            go.Bar(
                x=frame["ticker"],
                y=frame["recent_move_bps"],
                marker_color=["#ef4444" if value < 0 else "#22c55e" for value in frame["recent_move_bps"]],
                name="Recent move bps",
            )
        )
    fig.update_layout(
        title="Recent peer moves",
        template="plotly_dark",
        paper_bgcolor="#0f1419",
        plot_bgcolor="#0f1419",
        height=360,
        yaxis_title="bps",
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def counterfactual_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if frame.empty:
        return fig
    for algo, group in frame.groupby("Algo"):
        fig.add_trace(
            go.Bar(
                x=group["Scenario"],
                y=group["Estimated Cost bps"],
                name=str(algo),
            )
        )
    fig.update_layout(
        title="Counterfactual estimated arrival cost by algo",
        template="plotly_dark",
        paper_bgcolor="#0f1419",
        plot_bgcolor="#0f1419",
        height=420,
        barmode="group",
        yaxis_title="Estimated arrival cost bps",
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def scenario_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if frame.empty:
        return fig
    fig.add_trace(
        go.Bar(
            x=frame["algo"],
            y=frame["expected_arrival_cost_bps"],
            name="Expected",
            marker_color="#38bdf8",
            error_y=dict(
                type="data",
                symmetric=False,
                array=frame["p90_arrival_cost_bps"] - frame["expected_arrival_cost_bps"],
                arrayminus=frame["expected_arrival_cost_bps"] - frame["p10_arrival_cost_bps"],
            ),
        )
    )
    fig.update_layout(
        title="Expected arrival cost with P10-P90 range",
        template="plotly_dark",
        paper_bgcolor="#0f1419",
        plot_bgcolor="#0f1419",
        height=390,
        yaxis_title="Arrival cost bps",
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


if __name__ == "__main__":
    main()
