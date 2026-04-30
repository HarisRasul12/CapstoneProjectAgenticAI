from __future__ import annotations

from execlab.schedules import generate_twap_schedule
from execlab.simulator import run_cost_scenario_lab, simulate_fills
from tests.conftest import synthetic_bars


def test_buy_and_sell_bps_signs_flip_on_rising_market() -> None:
    bars = synthetic_bars(trend_bps=100)
    schedule = generate_twap_schedule(10_000, bars)

    buy = simulate_fills("TWAP", schedule, bars, side="buy", spread_bps=0, impact_bps_per_10pct=0)
    sell = simulate_fills("TWAP", schedule, bars, side="sell", spread_bps=0, impact_bps_per_10pct=0)

    assert buy.metrics.arrival_cost_bps > 0
    assert sell.metrics.arrival_cost_bps < 0
    assert buy.metrics.market_vwap > buy.metrics.arrival_price
    assert buy.metrics.total_quantity_executed == 10_000


def test_avg_fill_and_vwap_slippage_are_calculated() -> None:
    bars = synthetic_bars(trend_bps=0)
    schedule = generate_twap_schedule(10_000, bars)
    result = simulate_fills("TWAP", schedule, bars, side="buy", spread_bps=0, impact_bps_per_10pct=0)

    assert abs(result.metrics.avg_fill_price - result.metrics.market_vwap) < 0.10
    assert abs(result.metrics.vwap_slippage_bps) < 10


def test_scenario_lab_is_deterministic_with_fixed_seed() -> None:
    bars = synthetic_bars(trend_bps=50)
    schedule = generate_twap_schedule(10_000, bars)
    schedules = {"TWAP": schedule}

    first = run_cost_scenario_lab(schedules, bars, "buy", 2.0, 1.5, 10.0, 100, 123)
    second = run_cost_scenario_lab(schedules, bars, "buy", 2.0, 1.5, 10.0, 100, 123)

    assert first.model_dump() == second.model_dump()


def test_limit_buy_blocks_non_executable_fills_and_reports_unfilled() -> None:
    bars = synthetic_bars(trend_bps=100)
    schedule = generate_twap_schedule(10_000, bars)
    result = simulate_fills(
        "TWAP",
        schedule,
        bars,
        side="buy",
        spread_bps=0,
        impact_bps_per_10pct=0,
        limit_price=99.0,
    )

    assert result.metrics.total_quantity_executed == 0
    assert result.metrics.unfilled_quantity == 10_000
    assert result.metrics.completion_rate == 0
    assert all(not fill.executable for fill in result.fills)


def test_limit_sell_blocks_when_price_is_below_limit() -> None:
    bars = synthetic_bars(trend_bps=-100)
    schedule = generate_twap_schedule(10_000, bars)
    result = simulate_fills(
        "TWAP",
        schedule,
        bars,
        side="sell",
        spread_bps=0,
        impact_bps_per_10pct=0,
        limit_price=101.0,
    )

    assert result.metrics.total_quantity_executed == 0
    assert result.metrics.unfilled_quantity == 10_000
