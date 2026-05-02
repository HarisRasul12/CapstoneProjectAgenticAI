from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import date
from uuid import uuid4

import numpy as np
import pandas as pd

from execlab.agents import adk_is_available, create_custom_algo_planner_agent
from execlab.config import load_settings
from execlab.data import EASTERN
from execlab.schemas import BarRecord, CustomAlgoPlan, ExecutionRequest, MarketDataPayload
from execlab.service import ExecLabService


def main() -> None:
    settings = load_settings()
    if not settings.adk_enabled or not adk_is_available():
        raise SystemExit("ADK/Vertex smoke failed: ADK is not available from runtime settings.")

    plan = asyncio.run(_run_planner_smoke())
    payload = plan.model_dump(mode="json")
    print(
        "ADK_SMOKE_OK "
        + json.dumps(
            {
                "status": payload.get("status"),
                "style_hint": payload.get("style_hint"),
                "max_participation_rate": payload.get("max_participation_rate"),
                "completion_target_pct": payload.get("completion_target_pct"),
                "completion_target_time": payload.get("completion_target_time"),
            },
            sort_keys=True,
        )
    )


def full_main() -> None:
    settings = load_settings()
    if not settings.adk_enabled or not adk_is_available():
        raise SystemExit("Full ADK smoke failed: ADK is not available from runtime settings.")

    settings = replace(
        settings,
        require_adk_success=True,
        allow_transient_fallback=False,
        historical_curve_lookback_days=4,
        pretrade_lookback_sessions=4,
        beta_lookback_days=40,
        adk_timeout_seconds=max(settings.adk_timeout_seconds, 300.0),
    )
    request = ExecutionRequest(
        ticker="NVDA",
        trade_date=date(2026, 4, 20),
        side="buy",
        quantity=10_000,
        interval="5m",
        participation_rate=0.10,
        scenario_paths=50,
        custom_algo_instructions=(
            "PM wants 50% done by 11:00, max participation 10%, "
            "reduce exposure but avoid chasing liquidity."
        ),
    )
    result = ExecLabService(settings=settings, client=SmokeMarketDataClient()).run_backtest(request)
    if result.adk_status != "success":
        raise SystemExit(
            "Full ADK smoke failed: "
            f"status={result.adk_status}; error={result.adk_error_summary}; warnings={result.warnings}"
        )
    if not result.agent_reports:
        raise SystemExit("Full ADK smoke failed: no specialist agent reports returned.")
    if not result.memo.best_algo:
        raise SystemExit("Full ADK smoke failed: no final memo best_algo returned.")

    print(
        "ADK_FULL_SMOKE_OK "
        + json.dumps(
            {
                "adk_status": result.adk_status,
                "agent_report_count": len(result.agent_reports),
                "best_algo": result.memo.best_algo,
                "model": result.adk_model_used,
                "runtime_seconds": result.runtime_seconds,
            },
            sort_keys=True,
        )
    )


async def _run_planner_smoke() -> CustomAlgoPlan:
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai.types import Content, Part

    settings = load_settings()
    agent = create_custom_algo_planner_agent(settings)
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name=settings.app_name, session_service=session_service)
    session_id = f"adk-smoke-{uuid4().hex[:8]}"
    user_id = "cloud-smoke"

    await session_service.create_session(
        app_name=settings.app_name,
        user_id=user_id,
        session_id=session_id,
        state={
            "request": {
                "ticker": "NVDA",
                "side": "buy",
                "quantity": 10_000,
                "participation_rate": 0.10,
            },
            "custom_planner_context": {
                "user_desk_brief": (
                    "PM wants 50% done by 11:00, max participation 10%, "
                    "reduce exposure but avoid chasing liquidity."
                ),
                "market": {
                    "price_move_bps": 42.0,
                    "market_vwap": 125.10,
                    "arrival_price": 124.80,
                },
                "pretrade": {
                    "avg_spread_proxy_bps": 4.5,
                    "order_size_adv_pct": 0.01,
                    "current_vs_21d_volume": 1.05,
                },
                "planner_instruction": "Return a CustomAlgoPlan for this desk brief.",
            },
        },
    )
    message = Content(
        role="user",
        parts=[Part(text="Create a CustomAlgoPlan from the desk brief and context.")],
    )

    captured: dict[str, object] = {}
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=message,
    ):
        actions = getattr(event, "actions", None)
        delta = getattr(actions, "state_delta", None) if actions else None
        if isinstance(delta, dict):
            captured.update(delta)

    session = await session_service.get_session(
        app_name=settings.app_name,
        user_id=user_id,
        session_id=session_id,
    )
    if hasattr(session, "state") and isinstance(session.state, dict):
        captured.update(session.state)

    payload = captured.get("custom_algo_plan")
    if isinstance(payload, CustomAlgoPlan):
        return payload
    if isinstance(payload, dict):
        return CustomAlgoPlan.model_validate(payload)
    raise RuntimeError("ADK/Vertex smoke failed: CustomAlgoPlannerAgent returned no plan.")


class SmokeMarketDataClient:
    def __init__(self, trend_bps: float = 100.0):
        self.trend_bps = trend_bps

    def fetch_intraday_bars(
        self,
        ticker: str,
        trade_date: date,
        interval: str = "5m",
    ) -> tuple[pd.DataFrame, MarketDataPayload]:
        frame = _synthetic_bars(trade_date=trade_date, trend_bps=self._trend_for(ticker))
        payload = MarketDataPayload(
            ticker=ticker,
            trade_date=trade_date,
            interval=interval,
            provider="cloud-smoke-synthetic",
            row_count=len(frame),
            records=[
                BarRecord(
                    timestamp_et=row.timestamp_et.to_pydatetime(),
                    open=float(row.open),
                    high=float(row.high),
                    low=float(row.low),
                    close=float(row.close),
                    volume=float(row.volume),
                )
                for row in frame.itertuples(index=False)
            ],
        )
        return frame, payload

    def fetch_daily_history(
        self,
        ticker: str,
        end_date: date,
        lookback_days: int = 126,
    ) -> pd.DataFrame:
        dates = pd.bdate_range(end=end_date, periods=lookback_days).date
        seed = sum((idx + 1) * ord(char) for idx, char in enumerate(ticker.upper()))
        rng = np.random.default_rng(seed)
        market_factor = np.linspace(-0.006, 0.007, lookback_days)
        ticker_upper = ticker.upper()
        if ticker_upper == "SPY":
            returns = market_factor + rng.normal(0, 0.002, lookback_days)
        elif ticker_upper in {"XLK", "QQQ"}:
            returns = 1.15 * market_factor + rng.normal(0, 0.003, lookback_days)
        else:
            sector_tilt = 1.35 if ticker_upper in {"NVDA", "MSFT", "AAPL", "AMD"} else 0.85
            returns = sector_tilt * market_factor + rng.normal(0, 0.006, lookback_days)
        close = 100.0 * np.cumprod(1.0 + returns)
        return pd.DataFrame({"date": dates, "close": close})

    def _trend_for(self, ticker: str) -> float:
        ticker_upper = ticker.upper()
        if ticker_upper == "SPY":
            return 35.0
        if ticker_upper in {"XLK", "QQQ"}:
            return 55.0
        return self.trend_bps


def _synthetic_bars(
    trade_date: date,
    trend_bps: float,
    rows: int = 78,
) -> pd.DataFrame:
    timestamps = pd.date_range(
        f"{trade_date.isoformat()} 09:30",
        periods=rows,
        freq="5min",
        tz=EASTERN,
    )
    base = 100.0
    end = base * (1.0 + trend_bps / 10_000.0)
    close = np.linspace(base, end, rows)
    open_ = np.r_[base, close[:-1]]
    high = np.maximum(open_, close) + 0.04
    low = np.minimum(open_, close) - 0.04
    x = np.linspace(-1.0, 1.0, rows)
    volume = (70_000 + 50_000 * (x**2)).astype(int)
    return pd.DataFrame(
        {
            "timestamp_et": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


if __name__ == "__main__":
    main()
