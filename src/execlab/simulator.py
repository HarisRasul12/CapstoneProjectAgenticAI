from __future__ import annotations

import numpy as np
import pandas as pd

from execlab.analytics import bar_vwap_proxy, market_vwap
from execlab.schemas import (
    FillRecord,
    ScenarioAlgoResult,
    ScenarioReport,
    ScheduleSummary,
    SimulationResult,
    TcaMetrics,
)


def simulate_fills(
    algo: str,
    schedule: pd.DataFrame,
    bars: pd.DataFrame,
    side: str,
    spread_bps: float,
    impact_bps_per_10pct: float,
    limit_price: float | None = None,
    pov_mode: str = "strict_cap",
) -> SimulationResult:
    merged = schedule.merge(
        bars[["timestamp_et", "open", "high", "low", "close", "volume"]],
        on="timestamp_et",
        how="left",
    )
    fills: list[FillRecord] = []
    base_prices = bar_vwap_proxy(merged.rename(columns={"volume": "volume"}))
    side_sign = 1.0 if side == "buy" else -1.0

    carry_quantity = 0
    for idx, row in merged.iterrows():
        scheduled_qty = int(row.get("target_quantity", 0) or 0)
        qty = scheduled_qty + carry_quantity
        carry_quantity = 0
        if qty <= 0:
            continue
        bar_volume = float(row.get("volume", 0) or 0)
        participation = float(qty / bar_volume) if bar_volume > 0 else 0.0
        market_price = float(base_prices.iloc[idx])
        impact_bps = float(impact_bps_per_10pct) * (participation / 0.10) if participation > 0 else 0.0
        fill_price = market_price * (1.0 + side_sign * ((float(spread_bps) / 2.0 + impact_bps) / 10_000.0))
        executable = _limit_allows_fill(fill_price, side, limit_price)
        executed_quantity = qty if executable else 0
        blocked_reason = None if executable else f"Limit price {limit_price:.4f} blocked modeled fill."
        if not executable and pov_mode == "force_complete":
            carry_quantity = qty
        fills.append(
            FillRecord(
                timestamp_et=row["timestamp_et"].to_pydatetime()
                if hasattr(row["timestamp_et"], "to_pydatetime")
                else row["timestamp_et"],
                algo=algo,
                target_quantity=qty,
                executed_quantity=executed_quantity,
                unfilled_quantity=qty - executed_quantity,
                bar_volume=bar_volume,
                market_price=market_price,
                fill_price=float(fill_price),
                participation_rate=float(executed_quantity / bar_volume) if bar_volume > 0 else 0.0,
                executable=executable,
                blocked_reason=blocked_reason,
            )
        )

    metrics = calculate_tca_metrics(
        algo=algo,
        fills=fills,
        bars=bars,
        side=side,
    )
    summary = summarize_schedule(algo, schedule, fills)
    metrics = metrics.model_copy(
        update={
            "total_quantity_targeted": summary.parent_quantity,
            "unfilled_quantity": summary.unfilled_quantity,
            "completion_rate": summary.completion_rate,
            "cap_violation_count": summary.cap_violation_count,
        }
    )
    return SimulationResult(algo=algo, fills=fills, metrics=metrics, schedule_summary=summary)


def calculate_tca_metrics(algo: str, fills: list[FillRecord], bars: pd.DataFrame, side: str) -> TcaMetrics:
    if bars.empty:
        raise ValueError("Cannot calculate TCA metrics without market bars.")
    arrival = float(bars["open"].iloc[0])
    close = float(bars["close"].iloc[-1])
    vwap = market_vwap(bars)
    executed_qty = sum(fill.executed_quantity for fill in fills)
    targeted_qty = sum(fill.target_quantity for fill in fills)
    if executed_qty <= 0:
        avg_fill = 0.0
        max_participation = 0.0
    else:
        avg_fill = sum(fill.fill_price * fill.executed_quantity for fill in fills) / executed_qty
        max_participation = max((fill.participation_rate for fill in fills), default=0.0)
    unfilled_qty = max(0, targeted_qty - executed_qty)

    return TcaMetrics(
        algo=algo,
        avg_fill_price=float(avg_fill),
        arrival_price=arrival,
        market_vwap=vwap,
        close_price=close,
        arrival_cost_bps=_signed_cost_bps(avg_fill, arrival, side),
        vwap_slippage_bps=_signed_cost_bps(avg_fill, vwap, side),
        close_slippage_bps=_signed_cost_bps(avg_fill, close, side),
        total_quantity_targeted=int(targeted_qty),
        total_quantity_executed=int(executed_qty),
        unfilled_quantity=int(unfilled_qty),
        completion_rate=float(executed_qty / targeted_qty) if targeted_qty > 0 else 0.0,
        max_participation_rate=float(max_participation),
        cap_violation_count=0,
    )


def summarize_schedule(algo: str, schedule: pd.DataFrame, fills: list[FillRecord]) -> ScheduleSummary:
    child = schedule["target_quantity"].astype(int) if "target_quantity" in schedule else pd.Series(dtype=int)
    parent_quantity = int(schedule["parent_quantity"].iloc[0]) if "parent_quantity" in schedule and not schedule.empty else int(child.sum()) if not child.empty else 0
    executed = sum(fill.executed_quantity for fill in fills)
    cap_violation_count = int(schedule["cap_violation"].sum()) if "cap_violation" in schedule else 0
    return ScheduleSummary(
        algo=algo,
        parent_quantity=parent_quantity,
        total_quantity=int(child.sum()) if not child.empty else 0,
        child_order_count=int((child > 0).sum()) if not child.empty else 0,
        max_child_order=int(child.max()) if not child.empty else 0,
        max_participation_rate=float(max((fill.participation_rate for fill in fills), default=0.0)),
        unfilled_quantity=max(0, parent_quantity - int(executed)),
        completion_rate=float(executed / parent_quantity) if parent_quantity > 0 else 0.0,
        cap_violation_count=cap_violation_count,
        schedule_note=str(schedule["schedule_note"].iloc[0]) if "schedule_note" in schedule and not schedule.empty else "",
    )


def run_cost_scenario_lab(
    schedules: dict[str, pd.DataFrame],
    bars: pd.DataFrame,
    side: str,
    spread_bps: float,
    impact_bps_per_10pct: float,
    drift_bps_per_day: float,
    path_count: int,
    seed: int,
) -> ScenarioReport:
    rng = np.random.default_rng(seed)
    base_prices = bar_vwap_proxy(bars).to_numpy(dtype=float)
    close_returns = pd.Series(bars["close"]).pct_change().dropna().to_numpy(dtype=float)
    sigma = float(np.std(close_returns)) if close_returns.size else 0.0005
    bars_per_day = max(1, len(bars))
    drift_per_bar = (float(drift_bps_per_day) / 10_000.0) / bars_per_day
    side_sign = 1.0 if side == "buy" else -1.0
    arrival = float(bars["open"].iloc[0])
    results: list[ScenarioAlgoResult] = []

    for algo, schedule in schedules.items():
        qty = schedule["target_quantity"].to_numpy(dtype=float)
        volume = bars["volume"].to_numpy(dtype=float)
        costs: list[float] = []
        for _ in range(path_count):
            shocks = rng.normal(loc=drift_per_bar, scale=max(sigma, 0.0001), size=len(base_prices))
            scenario_prices = base_prices * np.exp(np.cumsum(shocks))
            participation = np.divide(qty, volume, out=np.zeros_like(qty), where=volume > 0)
            impact = impact_bps_per_10pct * (participation / 0.10)
            fill_prices = scenario_prices * (1.0 + side_sign * ((spread_bps / 2.0 + impact) / 10_000.0))
            total_qty = float(qty.sum())
            avg_fill = float(np.dot(fill_prices, qty) / total_qty) if total_qty > 0 else 0.0
            costs.append(_signed_cost_bps(avg_fill, arrival, side))

        arr = np.array(costs, dtype=float)
        results.append(
            ScenarioAlgoResult(
                algo=algo,
                expected_arrival_cost_bps=float(arr.mean()),
                p10_arrival_cost_bps=float(np.percentile(arr, 10)),
                p50_arrival_cost_bps=float(np.percentile(arr, 50)),
                p90_arrival_cost_bps=float(np.percentile(arr, 90)),
                probability_cost_positive=float((arr > 0).mean()),
            )
        )

    return ScenarioReport(
        path_count=path_count,
        seed=seed,
        spread_bps=spread_bps,
        impact_bps_per_10pct=impact_bps_per_10pct,
        drift_bps_per_day=drift_bps_per_day,
        results=sorted(results, key=lambda item: item.expected_arrival_cost_bps),
        caveats=[
            "Scenario Lab perturbs bar prices; it does not model order book queueing or venue routing.",
            "Spread is a proxy input, not observed NBBO.",
        ],
    )


def _signed_cost_bps(avg_fill: float, benchmark: float, side: str) -> float:
    if benchmark == 0 or avg_fill == 0:
        return 0.0
    if side == "buy":
        return float((avg_fill - benchmark) / benchmark * 10_000)
    return float((benchmark - avg_fill) / benchmark * 10_000)


def _limit_allows_fill(fill_price: float, side: str, limit_price: float | None) -> bool:
    if limit_price is None:
        return True
    if side == "buy":
        return fill_price <= limit_price
    return fill_price >= limit_price
