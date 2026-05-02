from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import date
from uuid import uuid4

import pandas as pd

from execlab.agents import adk_is_available, create_custom_algo_planner_agent, create_execlab_root_agent
from execlab.agentic_reports import build_counterfactual_report as make_counterfactual_report
from execlab.agentic_reports import build_debate_report as make_debate_report
from execlab.agentic_reports import build_execution_playbook as make_playbook_report
from execlab.analytics import build_historical_volume_curve as build_curve
from execlab.analytics import build_market_eda as build_eda
from execlab.causal import build_tca_causal_report as make_causal_report
from execlab.config import Settings, load_settings
from execlab.custom_algo import build_custom_algo_report as make_custom_algo_report
from execlab.data import MarketDataClient, MarketDataError, select_window_bars, validate_market_session
from execlab.schedules import (
    generate_is_schedule as make_is_schedule,
    generate_pov_schedule as make_pov_schedule,
    generate_twap_schedule as make_twap_schedule,
    generate_vwap_schedule as make_vwap_schedule,
)
from execlab.pretrade import (
    build_pretrade_analytics as make_pretrade_report,
    fetch_lookback_sessions,
    fit_expected_cost_model as make_expected_cost_report,
)
from execlab.risk import build_beta_risk_report as make_beta_risk_report
from execlab.risk import build_peer_stock_analysis as make_peer_report
from execlab.schemas import (
    AgentStepReport,
    BetaRiskReport,
    CounterfactualReport,
    CustomAlgoPlan,
    CustomAlgoReport,
    DebateReport,
    ExecutionPlaybookReport,
    ExpectedCostModelReport,
    ExecutionMemo,
    ExecutionRequest,
    MarketEDA,
    PeerStockAnalysisReport,
    PreTradeAnalyticsReport,
    RunResult,
    SimulationResult,
    TabNarrative,
    TabNarrativeBook,
    TcaCausalReport,
)
from execlab.simulator import run_cost_scenario_lab as run_scenario_lab
from execlab.simulator import simulate_fills as run_fill_simulation


@dataclass
class ToolRuntime:
    request: ExecutionRequest
    bars: pd.DataFrame = field(default_factory=pd.DataFrame)
    window_bars: pd.DataFrame = field(default_factory=pd.DataFrame)
    schedules: dict[str, pd.DataFrame] = field(default_factory=dict)
    simulations: dict[str, SimulationResult] = field(default_factory=dict)
    lookback_bars: pd.DataFrame = field(default_factory=pd.DataFrame)
    volume_curve: pd.Series = field(default_factory=pd.Series)
    volume_curve_source: str = "unavailable"
    eda: MarketEDA | None = None
    pretrade_report: PreTradeAnalyticsReport | None = None
    expected_cost_report: ExpectedCostModelReport | None = None
    beta_risk_report: BetaRiskReport | None = None
    peer_report: PeerStockAnalysisReport | None = None
    causal_report: TcaCausalReport | None = None
    debate_report: DebateReport | None = None
    counterfactual_report: CounterfactualReport | None = None
    playbook_report: ExecutionPlaybookReport | None = None
    custom_schedule: pd.DataFrame = field(default_factory=pd.DataFrame)
    custom_algo_plan: CustomAlgoPlan | None = None
    custom_algo_report: CustomAlgoReport | None = None
    scenario_report: object | None = None
    provider: str = "unknown"
    warnings: list[str] = field(default_factory=list)
    state_delta: dict[str, object] = field(default_factory=dict)
    agent_reports: dict[str, AgentStepReport] = field(default_factory=dict)
    agent_narratives: dict[str, TabNarrative] = field(default_factory=dict)
    adk_status: str = "not_attempted"
    adk_attempted: bool = False
    adk_error_summary: str | None = None
    adk_error_category: str | None = None
    adk_model_used: str | None = None
    execution_trace: list[dict[str, object]] = field(default_factory=list)


class ExecLabToolset:
    def __init__(self, settings: Settings, client: MarketDataClient, runtime: ToolRuntime):
        self.settings = settings
        self.client = client
        self.runtime = runtime

    def tools(self):
        return [
            self.fetch_intraday_bars,
            self.validate_market_session,
            self.build_market_eda,
            self.build_historical_volume_curve,
            self.build_pretrade_analytics,
            self.fit_expected_cost_model,
            self.build_beta_risk_model,
            self.build_peer_stock_analysis,
            self.build_tca_causal_analysis,
            self.build_debate_report,
            self.build_counterfactual_report,
            self.build_execution_playbook,
            self.build_custom_algo,
            self.analyze_limit_feasibility,
            self.generate_twap_schedule,
            self.generate_vwap_schedule,
            self.generate_pov_schedule,
            self.generate_is_schedule,
            self.simulate_fills,
            self.calculate_tca_metrics,
            self.run_cost_scenario_lab,
            self.get_execution_summary,
        ]

    def fetch_intraday_bars(
        self,
        ticker: str | None = None,
        trade_date: str | None = None,
        interval: str | None = None,
    ) -> dict:
        if not self.runtime.bars.empty:
            _trace(self.runtime, "tool.fetch_intraday_bars", details="cache hit")
            return self._market_payload()

        req = self.runtime.request
        use_date = date.fromisoformat(trade_date) if trade_date else req.trade_date
        started = time.perf_counter()
        frame, payload = self.client.fetch_intraday_bars(
            ticker=ticker or req.ticker,
            trade_date=use_date,
            interval=interval or req.interval,
        )
        self.runtime.provider = payload.provider
        self.runtime.bars = frame
        self.runtime.window_bars = select_window_bars(frame, req.start_time, req.end_time)
        self.runtime.warnings.extend(payload.warnings)
        self.runtime.warnings.extend(
            validate_market_session(self.runtime.window_bars, req.start_time, req.end_time)
        )
        if self.runtime.window_bars.empty:
            raise MarketDataError("No bars remain after applying the requested execution window.")
        _trace(
            self.runtime,
            "tool.fetch_intraday_bars",
            details=f"rows={len(frame)}, window_rows={len(self.runtime.window_bars)}",
            duration_ms=_elapsed_ms(started),
        )
        return self._market_payload()

    def validate_market_session(self) -> dict:
        self._ensure_bars()
        warnings = validate_market_session(
            self.runtime.window_bars,
            self.runtime.request.start_time,
            self.runtime.request.end_time,
        )
        self.runtime.warnings.extend(warnings)
        _trace(self.runtime, "tool.validate_market_session", details=f"warnings={len(warnings)}")
        return {"status": "ok" if not warnings else "warning", "warnings": warnings}

    def build_market_eda(self) -> dict:
        self._ensure_curve()
        self._ensure_eda()
        _trace(self.runtime, "tool.build_market_eda", details="eda ready")
        return self.runtime.eda.model_dump() if self.runtime.eda else {}

    def build_historical_volume_curve(self, lookback_days: int | None = None) -> dict:
        self._ensure_bars()
        if not self.runtime.volume_curve.empty:
            return {
                "source": self.runtime.volume_curve_source,
                "weights": self.runtime.volume_curve.round(6).tolist(),
            }
        req = self.runtime.request
        curve, source, warnings = build_curve(
            client=self.client,
            ticker=req.ticker,
            trade_date=req.trade_date,
            interval=req.interval,
            current_window_bars=self.runtime.window_bars,
            lookback_days=lookback_days or self.settings.historical_curve_lookback_days,
        )
        self.runtime.volume_curve = curve
        self.runtime.volume_curve_source = source
        self.runtime.warnings.extend(warnings)
        _trace(self.runtime, "tool.build_historical_volume_curve", details=source)
        return {"source": source, "weights": curve.round(6).tolist(), "warnings": warnings[:5]}

    def build_pretrade_analytics(self) -> dict:
        self._ensure_bars()
        if self.runtime.pretrade_report is None:
            req = self.runtime.request
            interval = "5m" if req.interval == "1m" else req.interval
            warnings: list[str] = []
            if interval != req.interval:
                warnings.append("21-day pre-trade analytics use 5m bars because 1m lookback is fragile with public intraday data.")
            lookback, fetch_warnings = fetch_lookback_sessions(
                client=self.client,
                ticker=req.ticker,
                anchor_date=req.trade_date,
                interval=interval,
                requested_sessions=self.settings.pretrade_lookback_sessions,
            )
            self.runtime.lookback_bars = lookback
            warnings.extend(fetch_warnings)
            self.runtime.pretrade_report = make_pretrade_report(
                request=req,
                current_window_bars=self.runtime.window_bars,
                lookback_bars=lookback,
                requested_sessions=self.settings.pretrade_lookback_sessions,
                warnings=warnings,
            )
            self.runtime.warnings.extend(self.runtime.pretrade_report.warnings)
        _trace(
            self.runtime,
            "tool.build_pretrade_analytics",
            details=f"sessions={self.runtime.pretrade_report.lookback_sessions if self.runtime.pretrade_report else 0}",
        )
        return self.runtime.pretrade_report.model_dump() if self.runtime.pretrade_report else {}

    def fit_expected_cost_model(self) -> dict:
        if self.runtime.pretrade_report is None:
            self.build_pretrade_analytics()
        if self.runtime.expected_cost_report is None:
            self.runtime.expected_cost_report = make_expected_cost_report(
                request=self.runtime.request,
                pretrade=self.runtime.pretrade_report,
                lookback_bars=self.runtime.lookback_bars,
                seed=self.runtime.request.seed,
            )
        _trace(
            self.runtime,
            "tool.fit_expected_cost_model",
            details=f"expected={self.runtime.expected_cost_report.expected_cost_bps:.2f}bps",
        )
        return self.runtime.expected_cost_report.model_dump()

    def build_beta_risk_model(self) -> dict:
        if self.runtime.beta_risk_report is None:
            self.runtime.beta_risk_report = make_beta_risk_report(
                request=self.runtime.request,
                client=self.client,
                lookback_days=self.settings.beta_lookback_days,
            )
            self.runtime.warnings.extend(self.runtime.beta_risk_report.warnings)
        _trace(
            self.runtime,
            "tool.build_beta_risk_model",
            details=(
                f"sector={self.runtime.beta_risk_report.sector_etf}, "
                f"beta={self.runtime.beta_risk_report.beta_market:.2f}"
            ),
        )
        return self.runtime.beta_risk_report.model_dump()

    def build_peer_stock_analysis(self) -> dict:
        self._ensure_beta_risk()
        if self.runtime.peer_report is None:
            self.runtime.peer_report = make_peer_report(
                request=self.runtime.request,
                client=self.client,
                beta_risk=self.runtime.beta_risk_report,
                lookback_days=self.settings.beta_lookback_days,
            )
            self.runtime.warnings.extend(self.runtime.peer_report.warnings)
        _trace(
            self.runtime,
            "tool.build_peer_stock_analysis",
            details=(
                f"peers={self.runtime.peer_report.analyzed_count}, "
                f"urgency={self.runtime.peer_report.urgency_recommendation}"
            ),
        )
        return self.runtime.peer_report.model_dump()

    def build_tca_causal_analysis(self) -> dict:
        self._ensure_eda()
        self._ensure_pretrade()
        self._ensure_beta_risk()
        self._ensure_peers()
        self._ensure_simulations()
        if self.runtime.causal_report is None:
            self.runtime.causal_report = make_causal_report(
                request=self.runtime.request,
                eda=self.runtime.eda,
                simulations=self.runtime.simulations,
                schedules=self.runtime.schedules,
                pretrade=self.runtime.pretrade_report,
                beta_risk=self.runtime.beta_risk_report,
                peer_report=self.runtime.peer_report,
            )
        _trace(
            self.runtime,
            "tool.build_tca_causal_analysis",
            details=f"best={self.runtime.causal_report.best_algo}",
        )
        return self.runtime.causal_report.model_dump()

    def build_debate_report(self) -> dict:
        self._ensure_pretrade()
        self._ensure_beta_risk()
        self._ensure_peers()
        self._ensure_simulations()
        if self.runtime.debate_report is None:
            self.runtime.debate_report = make_debate_report(
                request=self.runtime.request,
                simulations=self.runtime.simulations,
                schedules=self.runtime.schedules,
                pretrade=self.runtime.pretrade_report,
                beta_risk=self.runtime.beta_risk_report,
                peer_report=self.runtime.peer_report,
            )
        _trace(
            self.runtime,
            "tool.build_debate_report",
            details=f"judge={self.runtime.debate_report.judge_winner}, algo={self.runtime.debate_report.recommended_algo}",
        )
        return self.runtime.debate_report.model_dump()

    def build_counterfactual_report(self) -> dict:
        self._ensure_pretrade()
        self._ensure_peers()
        self._ensure_simulations()
        if self.runtime.counterfactual_report is None:
            self.runtime.counterfactual_report = make_counterfactual_report(
                request=self.runtime.request,
                simulations=self.runtime.simulations,
                schedules=self.runtime.schedules,
                pretrade=self.runtime.pretrade_report,
                peer_report=self.runtime.peer_report,
            )
        _trace(
            self.runtime,
            "tool.build_counterfactual_report",
            details=f"scenarios={len(self.runtime.counterfactual_report.scenarios)}",
        )
        return self.runtime.counterfactual_report.model_dump()

    def build_execution_playbook(self) -> dict:
        self._ensure_pretrade()
        self._ensure_beta_risk()
        self._ensure_peers()
        self._ensure_debate()
        self._ensure_counterfactuals()
        if self.runtime.playbook_report is None:
            self.runtime.playbook_report = make_playbook_report(
                request=self.runtime.request,
                debate=self.runtime.debate_report,
                counterfactuals=self.runtime.counterfactual_report,
                pretrade=self.runtime.pretrade_report,
                beta_risk=self.runtime.beta_risk_report,
                peer_report=self.runtime.peer_report,
            )
        _trace(
            self.runtime,
            "tool.build_execution_playbook",
            details=f"algo={self.runtime.playbook_report.recommended_algo}",
        )
        return self.runtime.playbook_report.model_dump()

    def build_custom_algo(self) -> dict:
        self._ensure_eda()
        self._ensure_pretrade()
        self._ensure_beta_risk()
        self._ensure_peers()
        self._ensure_debate()
        if self.runtime.custom_algo_report is None:
            schedule, report = make_custom_algo_report(
                request=self.runtime.request,
                bars=self.runtime.window_bars,
                volume_curve=self.runtime.volume_curve,
                eda=self.runtime.eda,
                pretrade=self.runtime.pretrade_report,
                beta_risk=self.runtime.beta_risk_report,
                peer_report=self.runtime.peer_report,
                debate_report=self.runtime.debate_report,
                custom_plan=self.runtime.custom_algo_plan,
            )
            self.runtime.custom_schedule = schedule
            self.runtime.custom_algo_report = report
        _trace(
            self.runtime,
            "tool.build_custom_algo",
            details=(
                f"{self.runtime.custom_algo_report.name}: "
                f"{self.runtime.custom_algo_report.simulation.metrics.arrival_cost_bps:.2f}bps"
            ),
        )
        return self.runtime.custom_algo_report.model_dump()

    def analyze_limit_feasibility(self) -> dict:
        self._ensure_simulations()
        limit = self.runtime.request.limit_price
        blocked = {
            algo: sum(1 for fill in sim.fills if not fill.executable)
            for algo, sim in self.runtime.simulations.items()
        }
        unfilled = {
            algo: sim.metrics.unfilled_quantity
            for algo, sim in self.runtime.simulations.items()
        }
        _trace(self.runtime, "tool.analyze_limit_feasibility", details=f"limit={limit}")
        return {
            "limit_price": limit,
            "blocked_fill_count_by_algo": blocked,
            "unfilled_quantity_by_algo": unfilled,
            "note": "Limit feasibility is based on modeled bar fills, not live order-book liquidity.",
        }

    def generate_twap_schedule(self) -> dict:
        self._ensure_bars()
        if "TWAP" not in self.runtime.schedules and "TWAP" in self.runtime.request.algos:
            self.runtime.schedules["TWAP"] = make_twap_schedule(
                self.runtime.request.quantity,
                self.runtime.window_bars,
            )
        return self._schedule_payload("TWAP")

    def generate_vwap_schedule(self) -> dict:
        self._ensure_curve()
        if "VWAP" not in self.runtime.schedules and "VWAP" in self.runtime.request.algos:
            self.runtime.schedules["VWAP"] = make_vwap_schedule(
                self.runtime.request.quantity,
                self.runtime.window_bars,
                self.runtime.volume_curve,
            )
        return self._schedule_payload("VWAP")

    def generate_pov_schedule(self) -> dict:
        self._ensure_bars()
        if "POV" not in self.runtime.schedules and "POV" in self.runtime.request.algos:
            self.runtime.schedules["POV"] = make_pov_schedule(
                self.runtime.request.quantity,
                self.runtime.window_bars,
                self.runtime.request.participation_rate,
                self.runtime.request.pov_mode,
            )
        return self._schedule_payload("POV")

    def generate_is_schedule(self) -> dict:
        self._ensure_bars()
        if "IS" not in self.runtime.schedules and "IS" in self.runtime.request.algos:
            self.runtime.schedules["IS"] = make_is_schedule(
                self.runtime.request.quantity,
                self.runtime.window_bars,
                self.runtime.request.urgency,
            )
        return self._schedule_payload("IS")

    def simulate_fills(self, algo: str | None = None) -> dict:
        self._ensure_schedules()
        algos = [algo] if algo else list(self.runtime.schedules.keys())
        for name in algos:
            if name in self.runtime.schedules and name not in self.runtime.simulations:
                self.runtime.simulations[name] = run_fill_simulation(
                    algo=name,
                    schedule=self.runtime.schedules[name],
                    bars=self.runtime.window_bars,
                    side=self.runtime.request.side,
                    spread_bps=self.runtime.request.spread_bps,
                    impact_bps_per_10pct=self.runtime.request.impact_bps_per_10pct,
                    limit_price=self.runtime.request.limit_price,
                    pov_mode=self.runtime.request.pov_mode,
                )
        _trace(self.runtime, "tool.simulate_fills", details=f"algos={','.join(algos)}")
        return self._simulation_payload()

    def calculate_tca_metrics(self) -> dict:
        self._ensure_simulations()
        _trace(self.runtime, "tool.calculate_tca_metrics", details="metrics ready")
        return {
            algo: sim.metrics.model_dump()
            for algo, sim in sorted(self.runtime.simulations.items())
        }

    def run_cost_scenario_lab(self) -> dict:
        self._ensure_schedules()
        if self.runtime.scenario_report is None:
            req = self.runtime.request
            self.runtime.scenario_report = run_scenario_lab(
                schedules=self.runtime.schedules,
                bars=self.runtime.window_bars,
                side=req.side,
                spread_bps=req.spread_bps,
                impact_bps_per_10pct=req.impact_bps_per_10pct,
                drift_bps_per_day=req.drift_bps_per_day,
                path_count=req.scenario_paths,
                seed=req.seed,
            )
        _trace(self.runtime, "tool.run_cost_scenario_lab", details="scenario report ready")
        return self.runtime.scenario_report.model_dump()

    def get_execution_summary(self) -> dict:
        self._ensure_eda()
        self._ensure_pretrade()
        self._ensure_beta_risk()
        self._ensure_peers()
        self._ensure_simulations()
        self.run_cost_scenario_lab()
        if self.runtime.causal_report is None:
            self.build_tca_causal_analysis()
        self._ensure_debate()
        self._ensure_counterfactuals()
        self._ensure_playbook()
        self._ensure_custom_algo()
        metrics = self.calculate_tca_metrics()
        best_arrival = min(
            metrics.values(),
            key=lambda item: item["arrival_cost_bps"],
        )["algo"]
        return {
            "request": self.runtime.request.model_dump(mode="json"),
            "eda": self.runtime.eda.model_dump() if self.runtime.eda else {},
            "metrics": metrics,
            "scenario_report": self.runtime.scenario_report.model_dump()
            if self.runtime.scenario_report
            else {},
            "pretrade_report": self.runtime.pretrade_report.model_dump()
            if self.runtime.pretrade_report
            else {},
            "expected_cost_report": self.runtime.expected_cost_report.model_dump()
            if self.runtime.expected_cost_report
            else {},
            "beta_risk_report": self.runtime.beta_risk_report.model_dump()
            if self.runtime.beta_risk_report
            else {},
            "peer_report": self.runtime.peer_report.model_dump()
            if self.runtime.peer_report
            else {},
            "causal_report": self.runtime.causal_report.model_dump()
            if self.runtime.causal_report
            else {},
            "debate_report": self.runtime.debate_report.model_dump()
            if self.runtime.debate_report
            else {},
            "counterfactual_report": self.runtime.counterfactual_report.model_dump()
            if self.runtime.counterfactual_report
            else {},
            "playbook_report": self.runtime.playbook_report.model_dump()
            if self.runtime.playbook_report
            else {},
            "custom_algo_report": self.runtime.custom_algo_report.model_dump()
            if self.runtime.custom_algo_report
            else {},
            "best_algo_by_arrival_cost": best_arrival,
            "warnings": _dedupe(self.runtime.warnings),
            "limitation": "This is a bar-based execution research simulator, not a production OMS/EMS backtester.",
        }

    def _ensure_bars(self) -> None:
        if self.runtime.bars.empty:
            self.fetch_intraday_bars()

    def _ensure_curve(self) -> None:
        self._ensure_bars()
        if self.runtime.volume_curve.empty:
            self.build_historical_volume_curve()

    def _ensure_eda(self) -> None:
        self._ensure_curve()
        if self.runtime.eda is None:
            req = self.runtime.request
            self.runtime.eda = build_eda(
                ticker=req.ticker,
                trade_date=req.trade_date,
                interval=req.interval,
                all_bars=self.runtime.bars,
                window_bars=self.runtime.window_bars,
                volume_curve_source=self.runtime.volume_curve_source,
            )

    def _ensure_pretrade(self) -> None:
        if self.runtime.pretrade_report is None:
            self.build_pretrade_analytics()
        if self.runtime.expected_cost_report is None:
            self.fit_expected_cost_model()

    def _ensure_beta_risk(self) -> None:
        if self.runtime.beta_risk_report is None:
            self.build_beta_risk_model()

    def _ensure_peers(self) -> None:
        self._ensure_beta_risk()
        if self.runtime.peer_report is None:
            self.build_peer_stock_analysis()

    def _ensure_debate(self) -> None:
        if self.runtime.debate_report is None:
            self.build_debate_report()

    def _ensure_counterfactuals(self) -> None:
        if self.runtime.counterfactual_report is None:
            self.build_counterfactual_report()

    def _ensure_playbook(self) -> None:
        if self.runtime.playbook_report is None:
            self.build_execution_playbook()

    def _ensure_custom_algo(self) -> None:
        if self.runtime.custom_algo_report is None:
            self.build_custom_algo()

    def _ensure_schedules(self) -> None:
        self._ensure_curve()
        if "TWAP" in self.runtime.request.algos:
            self.generate_twap_schedule()
        if "VWAP" in self.runtime.request.algos:
            self.generate_vwap_schedule()
        if "POV" in self.runtime.request.algos:
            self.generate_pov_schedule()
        if "IS" in self.runtime.request.algos:
            self.generate_is_schedule()

    def _ensure_simulations(self) -> None:
        self._ensure_schedules()
        if set(self.runtime.schedules) - set(self.runtime.simulations):
            self.simulate_fills()

    def _market_payload(self) -> dict:
        return {
            "provider": self.runtime.provider,
            "ticker": self.runtime.request.ticker,
            "trade_date": self.runtime.request.trade_date.isoformat(),
            "rows": len(self.runtime.bars),
            "window_rows": len(self.runtime.window_bars),
            "first_timestamp": str(self.runtime.window_bars["timestamp_et"].iloc[0])
            if not self.runtime.window_bars.empty
            else None,
            "last_timestamp": str(self.runtime.window_bars["timestamp_et"].iloc[-1])
            if not self.runtime.window_bars.empty
            else None,
            "warnings": _dedupe(self.runtime.warnings),
        }

    def _schedule_payload(self, algo: str) -> dict:
        frame = self.runtime.schedules.get(algo, pd.DataFrame())
        if frame.empty:
            return {"algo": algo, "status": "not_selected_or_empty"}
        return {
            "algo": algo,
            "status": "ok",
            "total_quantity": int(frame["target_quantity"].sum()),
            "child_order_count": int((frame["target_quantity"] > 0).sum()),
            "max_child_order": int(frame["target_quantity"].max()),
            "unfilled_quantity": max(
                0,
                int(frame["parent_quantity"].iloc[0]) - int(frame["target_quantity"].sum()),
            )
            if "parent_quantity" in frame
            else 0,
            "cap_violation_count": int(frame["cap_violation"].sum()) if "cap_violation" in frame else 0,
            "note": str(frame["schedule_note"].iloc[0]),
        }

    def _simulation_payload(self) -> dict:
        return {
            algo: {
                "avg_fill_price": sim.metrics.avg_fill_price,
                "arrival_cost_bps": sim.metrics.arrival_cost_bps,
                "vwap_slippage_bps": sim.metrics.vwap_slippage_bps,
                "close_slippage_bps": sim.metrics.close_slippage_bps,
            }
            for algo, sim in sorted(self.runtime.simulations.items())
        }


class ExecLabService:
    def __init__(self, settings: Settings | None = None, client: MarketDataClient | None = None):
        self.settings = settings or load_settings()
        self.client = client or MarketDataClient(self.settings)
        self._last_successful_model: str | None = None

    def run_backtest(self, request: ExecutionRequest) -> RunResult:
        started = time.perf_counter()
        runtime = ToolRuntime(request=request)
        toolset = ExecLabToolset(self.settings, self.client, runtime)
        adk_ready = self.settings.adk_enabled and adk_is_available()

        _trace(
            runtime,
            "run.request_received",
            details=f"ticker={request.ticker}, date={request.trade_date}, adk_ready={adk_ready}",
        )

        # Tool pass guarantees the UI has complete, inspectable outputs before agent synthesis.
        toolset.fetch_intraday_bars()
        toolset.build_historical_volume_curve()
        toolset.build_market_eda()
        toolset.build_pretrade_analytics()
        toolset.fit_expected_cost_model()
        toolset.build_beta_risk_model()
        toolset.build_peer_stock_analysis()
        toolset.generate_twap_schedule()
        toolset.generate_vwap_schedule()
        toolset.generate_pov_schedule()
        toolset.generate_is_schedule()
        toolset.simulate_fills()
        toolset.calculate_tca_metrics()
        toolset.run_cost_scenario_lab()
        toolset.build_tca_causal_analysis()
        toolset.build_debate_report()
        toolset.build_counterfactual_report()
        toolset.build_execution_playbook()
        if request.custom_algo_instructions and adk_ready:
            runtime.adk_attempted = True
            self._run_custom_planner_with_candidates(runtime)
        elif request.custom_algo_instructions and not adk_ready:
            runtime.warnings.append(
                "Custom algo chat brief was provided, but ADK/Vertex is unavailable. "
                "The custom schedule used market context only; no deterministic brief parser was used."
            )
        toolset.build_custom_algo()

        memo: ExecutionMemo | None = None
        if self.settings.require_adk_success and not adk_ready:
            runtime.adk_status = "adk_unavailable_fallback"
            runtime.adk_error_category = "config"
            runtime.adk_error_summary = (
                "ADK/Vertex is required for the full agent memo but is unavailable locally. "
                "Returning tool-grounded fallback memo."
            )
            runtime.warnings.append(runtime.adk_error_summary)
            _trace(runtime, "run.adk_unavailable", status="warning", details=runtime.adk_error_summary)
        elif adk_ready:
            runtime.adk_attempted = True
            self._run_adk_with_candidates(runtime, toolset)
            runtime.agent_reports = self._agent_reports_from_state(runtime.state_delta)
            runtime.agent_narratives = self._agent_narratives_from_state(runtime.state_delta)
            memo = self._memo_from_state(runtime.state_delta)

        if memo is None:
            memo = self._fallback_memo(toolset.get_execution_summary())
            if runtime.adk_status == "not_attempted":
                runtime.adk_status = "skipped"
            _trace(runtime, "run.memo_fallback", details="tool-grounded memo generated")
        else:
            _trace(runtime, "run.memo_adk", details="ADK memo generated")

        if (
            runtime.eda is None
            or runtime.scenario_report is None
            or runtime.pretrade_report is None
            or runtime.expected_cost_report is None
            or runtime.beta_risk_report is None
            or runtime.peer_report is None
            or runtime.causal_report is None
            or runtime.debate_report is None
            or runtime.counterfactual_report is None
            or runtime.playbook_report is None
            or runtime.custom_algo_report is None
        ):
            raise RuntimeError("Backtest failed to produce required tool outputs.")

        return RunResult(
            request=request,
            provider=runtime.provider,
            bars=runtime.bars,
            window_bars=runtime.window_bars,
            schedules=runtime.schedules,
            simulations=runtime.simulations,
            eda=runtime.eda,
            pretrade_report=runtime.pretrade_report,
            expected_cost_report=runtime.expected_cost_report,
            beta_risk_report=runtime.beta_risk_report,
            peer_report=runtime.peer_report,
            causal_report=runtime.causal_report,
            debate_report=runtime.debate_report,
            counterfactual_report=runtime.counterfactual_report,
            playbook_report=runtime.playbook_report,
            custom_schedule=runtime.custom_schedule,
            custom_algo_report=runtime.custom_algo_report,
            scenario_report=runtime.scenario_report,
            memo=memo,
            agent_reports=runtime.agent_reports,
            agent_narratives=runtime.agent_narratives,
            warnings=_dedupe(runtime.warnings),
            adk_status=runtime.adk_status,
            adk_attempted=runtime.adk_attempted,
            adk_error_summary=runtime.adk_error_summary,
            adk_model_used=runtime.adk_model_used,
            execution_trace=runtime.execution_trace,
            runtime_seconds=round(time.perf_counter() - started, 3),
        )

    def _run_adk_with_candidates(self, runtime: ToolRuntime, toolset: ExecLabToolset) -> None:
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", self.settings.gcp_region)
        if self.settings.gcp_project:
            os.environ.setdefault("GOOGLE_CLOUD_PROJECT", self.settings.gcp_project)

        candidates = [self._last_successful_model or self.settings.vertex_model]
        for candidate in self.settings.vertex_model_candidates:
            if candidate not in candidates:
                candidates.append(candidate)

        last_exc: Exception | None = None
        for model_name in candidates:
            try:
                self._run_adk_once(runtime, toolset, model_name=model_name)
                runtime.adk_status = "success"
                runtime.adk_model_used = model_name
                self._last_successful_model = model_name
                return
            except Exception as exc:  # pragma: no cover - network/credentials dependent
                last_exc = exc
                runtime.adk_error_summary = _flatten_exception_messages(exc)
                runtime.adk_error_category = _classify_adk_error(runtime.adk_error_summary)
                _trace(
                    runtime,
                    "run.adk_attempt_failed",
                    status="error",
                    details=f"{model_name}: {runtime.adk_error_category}",
                )
                if runtime.adk_error_category in {"network", "rate_limit"}:
                    break

        runtime.adk_status = "adk_error_fallback"
        runtime.adk_error_summary = runtime.adk_error_summary or str(last_exc or "Unknown ADK failure")
        runtime.warnings.append(f"ADK memo failed; tool-grounded memo returned. Detail: {runtime.adk_error_summary}")

    def _run_custom_planner_with_candidates(self, runtime: ToolRuntime) -> None:
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", self.settings.gcp_region)
        if self.settings.gcp_project:
            os.environ.setdefault("GOOGLE_CLOUD_PROJECT", self.settings.gcp_project)

        candidates = [self._last_successful_model or self.settings.vertex_model]
        for candidate in self.settings.vertex_model_candidates:
            if candidate not in candidates:
                candidates.append(candidate)

        last_exc: Exception | None = None
        for model_name in candidates:
            try:
                state = self._run_custom_planner_once(runtime, model_name=model_name)
                plan = self._custom_plan_from_state(state)
                if plan is not None:
                    runtime.custom_algo_plan = plan
                    runtime.adk_model_used = model_name
                    self._last_successful_model = model_name
                    _trace(
                        runtime,
                        "run.custom_algo_planner",
                        details=f"status={plan.status}, style={plan.style_hint}",
                    )
                    return
            except Exception as exc:  # pragma: no cover - network/credentials dependent
                last_exc = exc
                runtime.adk_error_summary = _flatten_exception_messages(exc)
                runtime.adk_error_category = _classify_adk_error(runtime.adk_error_summary)
                _trace(
                    runtime,
                    "run.custom_algo_planner_failed",
                    status="error",
                    details=f"{model_name}: {runtime.adk_error_category}",
                )
                if runtime.adk_error_category in {"network", "rate_limit"}:
                    break

        detail = runtime.adk_error_summary or str(last_exc or "Unknown custom planner failure")
        runtime.warnings.append(
            "CustomAlgoPlannerAgent failed; custom chat brief was not interpreted. "
            f"Detail: {detail}"
        )

    def _run_custom_planner_once(self, runtime: ToolRuntime, model_name: str) -> dict[str, object]:
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai.types import Content, Part

        agent = create_custom_algo_planner_agent(self.settings, model_name=model_name)
        session_service = InMemorySessionService()
        runner = Runner(agent=agent, app_name=self.settings.app_name, session_service=session_service)
        session_id = f"custom-planner-{uuid4().hex[:8]}"
        planner_context = self._build_custom_planner_context(runtime)

        async def _drive() -> dict[str, object]:
            await session_service.create_session(
                app_name=self.settings.app_name,
                user_id=runtime.request.user_id,
                session_id=session_id,
                state={
                    "request": runtime.request.model_dump(mode="json"),
                    "custom_planner_context": planner_context,
                },
            )
            message = Content(
                role="user",
                parts=[
                    Part(
                        text=(
                            "Create a CustomAlgoPlan from the user's desk brief and the "
                            "custom_planner_context. Return only the structured schema."
                        )
                    )
                ],
            )
            captured: dict[str, object] = {}
            async for event in runner.run_async(
                user_id=runtime.request.user_id,
                session_id=session_id,
                new_message=message,
            ):
                actions = getattr(event, "actions", None)
                delta = getattr(actions, "state_delta", None) if actions else None
                if isinstance(delta, dict):
                    captured.update(delta)
            session = await session_service.get_session(
                app_name=self.settings.app_name,
                user_id=runtime.request.user_id,
                session_id=session_id,
            )
            if hasattr(session, "state") and isinstance(session.state, dict):
                captured.update(session.state)
            return captured

        planner_timeout = min(max(self.settings.adk_timeout_seconds / 2.0, 60.0), 180.0)
        return asyncio.run(asyncio.wait_for(_drive(), timeout=planner_timeout))

    def _run_adk_once(self, runtime: ToolRuntime, toolset: ExecLabToolset, model_name: str) -> None:
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai.types import Content, Part

        agent = create_execlab_root_agent(self.settings, model_name=model_name)
        session_service = InMemorySessionService()
        runner = Runner(agent=agent, app_name=self.settings.app_name, session_service=session_service)
        session_id = f"execlab-{uuid4().hex[:8]}"
        execution_context = self._build_agent_execution_context(runtime, toolset)

        async def _drive() -> dict[str, object]:
            await session_service.create_session(
                app_name=self.settings.app_name,
                user_id=runtime.request.user_id,
                session_id=session_id,
                state={
                    "request": runtime.request.model_dump(mode="json"),
                    "execution_context": execution_context,
                },
            )
            message = Content(
                role="user",
                parts=[
                    Part(
                        text=(
                            "Run the ExecLab AI synthesis handoff using the precomputed execution_context. "
                            "Do not call tools. Each specialist agent should summarize its section, then "
                            "produce the final execution recommendation memo."
                        )
                    )
                ],
            )
            captured: dict[str, object] = {}
            async for event in runner.run_async(
                user_id=runtime.request.user_id,
                session_id=session_id,
                new_message=message,
            ):
                actions = getattr(event, "actions", None)
                delta = getattr(actions, "state_delta", None) if actions else None
                if isinstance(delta, dict):
                    captured.update(delta)
            session = await session_service.get_session(
                app_name=self.settings.app_name,
                user_id=runtime.request.user_id,
                session_id=session_id,
            )
            if hasattr(session, "state") and isinstance(session.state, dict):
                captured.update(session.state)
            return captured

        runtime.state_delta = asyncio.run(
            asyncio.wait_for(_drive(), timeout=self.settings.adk_timeout_seconds)
        )

    @staticmethod
    def _build_custom_planner_context(runtime: ToolRuntime) -> dict[str, object]:
        metrics = {
            algo: sim.metrics.model_dump(mode="json")
            for algo, sim in sorted(runtime.simulations.items())
        }
        schedule_summaries = {
            algo: sim.schedule_summary.model_dump(mode="json")
            for algo, sim in sorted(runtime.simulations.items())
        }
        return {
            "request": runtime.request.model_dump(mode="json"),
            "user_desk_brief": runtime.request.custom_algo_instructions,
            "market": runtime.eda.model_dump(mode="json") if runtime.eda else {},
            "pretrade": runtime.pretrade_report.model_dump(mode="json")
            if runtime.pretrade_report
            else {},
            "expected_cost_model": runtime.expected_cost_report.model_dump(mode="json")
            if runtime.expected_cost_report
            else {},
            "beta_risk": runtime.beta_risk_report.model_dump(mode="json")
            if runtime.beta_risk_report
            else {},
            "peer_analysis": runtime.peer_report.model_dump(mode="json")
            if runtime.peer_report
            else {},
            "causal_tca": runtime.causal_report.model_dump(mode="json")
            if runtime.causal_report
            else {},
            "agent_debate": runtime.debate_report.model_dump(mode="json")
            if runtime.debate_report
            else {},
            "execution_playbook": runtime.playbook_report.model_dump(mode="json")
            if runtime.playbook_report
            else {},
            "algo_metrics": metrics,
            "schedule_summaries": schedule_summaries,
            "warnings": _dedupe(runtime.warnings)[:8],
            "planner_instruction": (
                "Infer the user's custom execution constraints. Prefer asking follow-up questions "
                "only when the brief is impossible to translate into a safe educational plan. "
                "Do not invent numbers; use null when the user did not specify a hard cap or deadline."
            ),
        }

    @staticmethod
    def _build_agent_execution_context(runtime: ToolRuntime, toolset: ExecLabToolset) -> dict[str, object]:
        summary = toolset.get_execution_summary()
        pretrade = summary.get("pretrade_report", {}) if isinstance(summary, dict) else {}
        expected = summary.get("expected_cost_report", {}) if isinstance(summary, dict) else {}
        beta_risk = summary.get("beta_risk_report", {}) if isinstance(summary, dict) else {}
        peer = summary.get("peer_report", {}) if isinstance(summary, dict) else {}
        causal = summary.get("causal_report", {}) if isinstance(summary, dict) else {}
        debate = summary.get("debate_report", {}) if isinstance(summary, dict) else {}
        counterfactual = summary.get("counterfactual_report", {}) if isinstance(summary, dict) else {}
        playbook = summary.get("playbook_report", {}) if isinstance(summary, dict) else {}
        custom = summary.get("custom_algo_report", {}) if isinstance(summary, dict) else {}
        scenario = summary.get("scenario_report", {}) if isinstance(summary, dict) else {}
        metrics = summary.get("metrics", {}) if isinstance(summary, dict) else {}
        eda = summary.get("eda", {}) if isinstance(summary, dict) else {}
        cost_breakdown = expected.get("cost_breakdown", []) if isinstance(expected, dict) else []
        coefficients = expected.get("coefficients", []) if isinstance(expected, dict) else []
        scenario_results = scenario.get("results", []) if isinstance(scenario, dict) else []
        index_points = beta_risk.get("index_comparison", []) if isinstance(beta_risk, dict) else []
        final_index_point = index_points[-1] if index_points else {}
        return {
            "request": runtime.request.model_dump(mode="json"),
            "market": {
                "arrival_price": eda.get("arrival_price"),
                "market_vwap": eda.get("market_vwap"),
                "close_price": eda.get("close_price"),
                "price_move_bps": eda.get("price_move_bps"),
                "realized_volatility_bps": eda.get("realized_volatility_bps"),
                "window_volume": eda.get("window_volume"),
                "volume_curve_source": eda.get("volume_curve_source"),
            },
            "pretrade": {
                "lookback_sessions": pretrade.get("lookback_sessions"),
                "adv_shares": pretrade.get("adv_shares"),
                "order_size_adv_pct": pretrade.get("order_size_adv_pct"),
                "avg_spread_proxy_bps": pretrade.get("avg_spread_proxy_bps"),
                "avg_volatility_bps": pretrade.get("avg_volatility_bps"),
                "time_risk_score": pretrade.get("time_risk_score"),
                "current_vs_21d_volume": pretrade.get("current_vs_21d_volume"),
                "warnings": pretrade.get("warnings", [])[:6],
            },
            "expected_cost_model": {
                "expected_cost_bps": expected.get("expected_cost_bps"),
                "model_r2": expected.get("model_r2"),
                "observation_count": expected.get("observation_count"),
                "cost_breakdown": cost_breakdown[:8],
                "coefficients": coefficients[:8],
                "caveats": expected.get("caveats", [])[:5] if isinstance(expected, dict) else [],
            },
            "beta_risk": {
                "market_etf": beta_risk.get("market_etf"),
                "sector_etf": beta_risk.get("sector_etf"),
                "sector_label": beta_risk.get("sector_label"),
                "mapping_confidence": beta_risk.get("mapping_confidence"),
                "mapping_reason": beta_risk.get("mapping_reason"),
                "observation_count": beta_risk.get("observation_count"),
                "beta_market": beta_risk.get("beta_market"),
                "beta_sector": beta_risk.get("beta_sector"),
                "correlation_market": beta_risk.get("correlation_market"),
                "correlation_sector": beta_risk.get("correlation_sector"),
                "r_squared": beta_risk.get("r_squared"),
                "systematic_risk_bps": beta_risk.get("systematic_risk_bps"),
                "sector_risk_bps": beta_risk.get("sector_risk_bps"),
                "idiosyncratic_risk_bps": beta_risk.get("idiosyncratic_risk_bps"),
                "total_timing_risk_bps": beta_risk.get("total_timing_risk_bps"),
                "intraday_final_move": final_index_point,
                "warnings": beta_risk.get("warnings", [])[:5] if isinstance(beta_risk, dict) else [],
            },
            "peer_analysis": {
                "sector_etf": peer.get("sector_etf"),
                "candidate_count": peer.get("candidate_count"),
                "analyzed_count": peer.get("analyzed_count"),
                "average_peer_correlation": peer.get("average_peer_correlation"),
                "crowding_score": peer.get("crowding_score"),
                "median_peer_move_bps": peer.get("median_peer_move_bps"),
                "target_recent_move_bps": peer.get("target_recent_move_bps"),
                "urgency_recommendation": peer.get("urgency_recommendation"),
                "rationale": peer.get("rationale"),
                "market_impact_note": peer.get("market_impact_note"),
                "peers": peer.get("peers", [])[:6] if isinstance(peer, dict) else [],
                "warnings": peer.get("warnings", [])[:5] if isinstance(peer, dict) else [],
            },
            "causal_tca": {
                "headline": causal.get("headline"),
                "best_algo": causal.get("best_algo"),
                "bullets": causal.get("bullets", [])[:6] if isinstance(causal, dict) else [],
                "caveats": causal.get("caveats", [])[:4] if isinstance(causal, dict) else [],
            },
            "agent_debate": {
                "fast_case": debate.get("fast_case"),
                "liquidity_case": debate.get("liquidity_case"),
                "judge_winner": debate.get("judge_winner"),
                "recommended_algo": debate.get("recommended_algo"),
                "confidence": debate.get("confidence"),
                "deciding_factors": debate.get("deciding_factors", [])[:5] if isinstance(debate, dict) else [],
                "judge_rationale": debate.get("judge_rationale"),
            },
            "counterfactuals": {
                "base_winner": counterfactual.get("base_winner"),
                "summary": counterfactual.get("summary"),
                "scenarios": counterfactual.get("scenarios", [])[:6] if isinstance(counterfactual, dict) else [],
            },
            "execution_playbook": {
                "recommended_algo": playbook.get("recommended_algo"),
                "urgency": playbook.get("urgency"),
                "participation_guidance": playbook.get("participation_guidance"),
                "limit_guidance": playbook.get("limit_guidance"),
                "monitoring_triggers": playbook.get("monitoring_triggers", [])[:5] if isinstance(playbook, dict) else [],
                "switch_rules": playbook.get("switch_rules", [])[:5] if isinstance(playbook, dict) else [],
                "rationale": playbook.get("rationale"),
            },
            "custom_algo": {
                "name": custom.get("name"),
                "style": custom.get("style"),
                "description": custom.get("description"),
                "components": custom.get("components", [])[:6] if isinstance(custom, dict) else [],
                "parameters": custom.get("parameters", {}) if isinstance(custom, dict) else {},
                "rationale": custom.get("rationale", [])[:6] if isinstance(custom, dict) else [],
                "metrics": (
                    custom.get("simulation", {}).get("metrics", {})
                    if isinstance(custom.get("simulation", {}), dict)
                    else {}
                )
                if isinstance(custom, dict)
                else {},
                "caveats": custom.get("caveats", [])[:4] if isinstance(custom, dict) else [],
            },
            "custom_algo_agent_plan": runtime.custom_algo_plan.model_dump(mode="json")
            if runtime.custom_algo_plan
            else {},
            "algo_metrics": metrics,
            "schedule_summaries": {
                algo: sim.schedule_summary.model_dump(mode="json")
                for algo, sim in sorted(runtime.simulations.items())
            },
            "scenario_results": scenario_results[:6],
            "limit_feasibility": toolset.analyze_limit_feasibility(),
            "warnings": summary.get("warnings", [])[:8],
            "limitation": summary.get(
                "limitation",
                "This is a bar-based execution research simulator, not a production OMS/EMS backtester.",
            ),
        }

    @staticmethod
    def _custom_plan_from_state(state: dict[str, object]) -> CustomAlgoPlan | None:
        payload = state.get("custom_algo_plan")
        if isinstance(payload, CustomAlgoPlan):
            return payload
        if isinstance(payload, dict):
            try:
                return CustomAlgoPlan.model_validate(payload)
            except Exception:
                return None
        return None

    @staticmethod
    def _memo_from_state(state: dict[str, object]) -> ExecutionMemo | None:
        payload = state.get("execution_memo") or state.get("execution_memo_draft")
        if isinstance(payload, ExecutionMemo):
            return payload
        if isinstance(payload, dict):
            try:
                return ExecutionMemo.model_validate(payload)
            except Exception:
                return None
        return None

    @staticmethod
    def _agent_reports_from_state(state: dict[str, object]) -> dict[str, AgentStepReport]:
        report_keys = [
            "market_data_report",
            "volume_curve_report",
            "pretrade_report_agent",
            "expected_cost_report_agent",
            "historical_regression_report",
            "beta_risk_report_agent",
            "peer_cluster_report_agent",
            "strategy_report",
            "simulation_report",
            "tca_report",
            "cause_effect_report",
            "fast_execution_argument",
            "liquidity_seeking_argument",
            "debate_judge_report",
            "counterfactual_report_agent",
            "playbook_report_agent",
            "custom_algo_designer_report",
            "tab_insight_report_agent",
            "limit_feasibility_report",
        ]
        reports: dict[str, AgentStepReport] = {}
        for key in report_keys:
            payload = state.get(key)
            if isinstance(payload, AgentStepReport):
                reports[key] = payload
            elif isinstance(payload, dict):
                try:
                    reports[key] = AgentStepReport.model_validate(payload)
                except Exception:
                    continue
        return reports

    @staticmethod
    def _agent_narratives_from_state(state: dict[str, object]) -> dict[str, TabNarrative]:
        payload = state.get("tab_narratives")
        book: TabNarrativeBook | None = None
        if isinstance(payload, TabNarrativeBook):
            book = payload
        elif isinstance(payload, dict):
            try:
                book = TabNarrativeBook.model_validate(payload)
            except Exception:
                book = None
        if book is None:
            return {}
        return {
            narrative.tab_key: narrative
            for narrative in book.narratives
            if narrative.tab_key
        }

    @staticmethod
    def _fallback_memo(summary: dict) -> ExecutionMemo:
        metrics = summary.get("metrics", {})
        if metrics:
            ranked = sorted(metrics.values(), key=lambda item: item["arrival_cost_bps"])
            best = ranked[0]
            worst = ranked[-1]
            best_algo = str(best["algo"])
            thesis = (
                f"{best_algo} had the lowest arrival-cost result in this bar-based backtest. "
                f"It posted {best['arrival_cost_bps']:.2f} bps versus arrival, while "
                f"{worst['algo']} was weakest at {worst['arrival_cost_bps']:.2f} bps."
            )
            evidence = [
                (
                    f"{item['algo']}: avg fill {item['avg_fill_price']:.4f}, "
                    f"arrival {item['arrival_cost_bps']:.2f} bps, "
                    f"VWAP {item['vwap_slippage_bps']:.2f} bps, "
                    f"close {item['close_slippage_bps']:.2f} bps."
                )
                for item in ranked
            ]
        else:
            best_algo = "Unavailable"
            thesis = "No completed algorithm metrics were available."
            evidence = []

        scenario = summary.get("scenario_report", {})
        scenario_results = scenario.get("results", []) if isinstance(scenario, dict) else []
        scenario_text = ""
        if scenario_results:
            leader = min(scenario_results, key=lambda item: item["expected_arrival_cost_bps"])
            scenario_text = (
                f"In Scenario Lab, {leader['algo']} had the lowest expected arrival cost "
                f"at {leader['expected_arrival_cost_bps']:.2f} bps using the configured "
                "spread, drift, and impact assumptions."
            )

        caveats = list(summary.get("warnings", []))[:4]
        expected = summary.get("expected_cost_report", {})
        if isinstance(expected, dict) and expected.get("expected_cost_bps") is not None:
            evidence.append(
                f"Pre-trade expected cost model estimate: {expected['expected_cost_bps']:.2f} bps."
            )
        beta_risk = summary.get("beta_risk_report", {})
        if isinstance(beta_risk, dict) and beta_risk.get("observation_count", 0) > 0:
            evidence.append(
                f"Beta risk map: {beta_risk['ticker']} vs {beta_risk['market_etf']} and "
                f"{beta_risk['sector_etf']} ({beta_risk['sector_label']}); "
                f"market beta {beta_risk['beta_market']:.2f}, sector beta "
                f"{beta_risk['beta_sector']:.2f}, idiosyncratic timing risk "
                f"{beta_risk['idiosyncratic_risk_bps']:.2f} bps."
            )
        peer = summary.get("peer_report", {})
        if isinstance(peer, dict) and peer.get("analyzed_count", 0) > 0:
            evidence.append(
                f"Peer cluster: {peer['analyzed_count']} peers in {peer['sector_etf']}; "
                f"crowding score {peer['crowding_score']:.2f}, median peer move "
                f"{peer['median_peer_move_bps']:.2f} bps, recommendation: "
                f"{peer['urgency_recommendation']}."
            )
        debate = summary.get("debate_report", {})
        if isinstance(debate, dict) and debate.get("recommended_algo"):
            evidence.append(
                f"Agent debate: {debate['judge_winner']} won with {debate['confidence']:.0%} "
                f"confidence; recommended algo {debate['recommended_algo']}."
            )
        custom = summary.get("custom_algo_report", {})
        if isinstance(custom, dict) and custom.get("simulation"):
            custom_metrics = custom["simulation"].get("metrics", {})
            if isinstance(custom_metrics, dict):
                evidence.append(
                    f"Custom algo designer: {custom.get('name')} ({custom.get('style')}) "
                    f"modeled at {custom_metrics.get('arrival_cost_bps', 0):.2f} bps arrival cost "
                    f"with {custom_metrics.get('completion_rate', 0) * 100:.1f}% completion."
                )
        counterfactual = summary.get("counterfactual_report", {})
        if isinstance(counterfactual, dict) and counterfactual.get("summary"):
            evidence.append(f"Counterfactual robustness: {counterfactual['summary']}")
        causal = summary.get("causal_report", {})
        if isinstance(causal, dict):
            for bullet in causal.get("bullets", [])[:3]:
                evidence.append(
                    f"{bullet['driver']}: {bullet['evidence']} Implication: {bullet['implication']}"
                )
        caveats.extend(
            [
                "Public OHLCV bars cannot model queue position, venue selection, hidden liquidity, or adverse selection.",
                "Spread is a proxy input, not observed NBBO.",
            ]
        )
        return ExecutionMemo(
            best_algo=best_algo,
            thesis=thesis,
            evidence=evidence,
            caveats=_dedupe(caveats),
            scenario_interpretation=scenario_text,
        )


def _trace(
    runtime: ToolRuntime,
    step: str,
    status: str = "ok",
    details: str = "",
    duration_ms: float | None = None,
) -> None:
    payload: dict[str, object] = {"step": step, "status": status}
    if details:
        payload["details"] = details
    if duration_ms is not None:
        payload["duration_ms"] = round(duration_ms, 2)
    runtime.execution_trace.append(payload)


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _flatten_exception_messages(exc: Exception) -> str:
    messages: list[str] = []
    cursor: BaseException | None = exc
    while cursor is not None:
        messages.append(str(cursor))
        cursor = cursor.__cause__ or cursor.__context__
    return " | ".join(msg for msg in messages if msg)


def _classify_adk_error(message: str) -> str:
    text = (message or "").lower()
    if any(token in text for token in ["deadline", "timeout", "connection", "dns", "network"]):
        return "network"
    if any(token in text for token in ["quota", "rate", "429", "resource exhausted"]):
        return "rate_limit"
    if any(token in text for token in ["model", "not found", "permission", "credentials"]):
        return "model_config"
    if any(token in text for token in ["schema", "validation"]):
        return "schema"
    return "unknown"
