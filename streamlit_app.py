from __future__ import annotations

from datetime import date, datetime, time, timedelta
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

    tabs = st.tabs(["Pre-Trade Lab", "Risk Model", "Peers", "TCA", "Charts", "Scenario Lab", "Agent Memo", "Data Room"])
    with tabs[0]:
        render_pretrade_lab(result)

    with tabs[1]:
        render_risk_model(result)

    with tabs[2]:
        render_peer_analysis(result)

    with tabs[3]:
        st.subheader("TCA comparison")
        agent_note(
            "What this tab means",
            "BenchmarkTcaAgent, CauseEffectTcaAgent",
            "TCA compares each schedule against arrival price, market VWAP, and close. "
            "For the selected side, lower bps is better; positive bps means the execution was worse than the benchmark.",
        )
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

    with tabs[4]:
        agent_note(
            "What this tab means",
            "AlgoStrategyAgent, ExecutionSimulatorAgent, LimitFeasibilityAgent",
            "These charts show how each pseudo execution agent would trade through the day, where fills occur, "
            "and whether participation or limit rules block shares.",
        )
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(price_and_fills_chart(result), use_container_width=True)
        with c2:
            st.plotly_chart(schedule_vs_volume_chart(result), use_container_width=True)
        st.plotly_chart(participation_chart(result), use_container_width=True)
        st.plotly_chart(cumulative_completion_chart(result), use_container_width=True)

    with tabs[5]:
        st.subheader("Cost scenario lab")
        agent_note(
            "What this tab means",
            "ExpectedCostModelAgent, HistoricalRegressionAgent",
            "The scenario lab perturbs spread, drift, and impact assumptions. It is an expected-cost stress test, "
            "not a venue-level fill simulator.",
        )
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

    with tabs[6]:
        memo = result.memo
        agent_note(
            "What this tab means",
            "NarrativeExplanationAgent, CriticGoldenSetAgent",
            "The final agents consume the computed tool outputs, peer cluster report, beta risk map, and TCA bullets "
            "to write a recommendation memo.",
        )
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

    with tabs[7]:
        st.subheader("Fills")
        agent_note(
            "What this tab means",
            "All deterministic tools",
            "This is the audit trail: modeled fill rows, blocked fills, and the sequence of tool calls used by the agents.",
        )
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


def fills_dataframe(result) -> pd.DataFrame:
    rows = []
    for sim in result.simulations.values():
        for fill in sim.fills:
            rows.append(fill.model_dump())
    return pd.DataFrame(rows)


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
    colors = {"TWAP": "#38bdf8", "VWAP": "#22c55e", "POV": "#f59e0b", "IS": "#ef4444"}
    for algo, sim in result.simulations.items():
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
    colors = {"TWAP": "#38bdf8", "VWAP": "#22c55e", "POV": "#f59e0b", "IS": "#ef4444"}
    for algo, schedule in result.schedules.items():
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


def participation_chart(result) -> go.Figure:
    fig = go.Figure()
    colors = {"TWAP": "#38bdf8", "VWAP": "#22c55e", "POV": "#f59e0b", "IS": "#ef4444"}
    for algo, sim in result.simulations.items():
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
    colors = {"TWAP": "#38bdf8", "VWAP": "#22c55e", "POV": "#f59e0b", "IS": "#ef4444"}
    for algo, sim in result.simulations.items():
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
