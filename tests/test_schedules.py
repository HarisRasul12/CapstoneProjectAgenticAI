from __future__ import annotations

from execlab.analytics import build_historical_volume_curve
from execlab.config import Settings
from execlab.schedules import (
    generate_is_schedule,
    generate_pov_schedule,
    generate_twap_schedule,
    generate_vwap_schedule,
)
from tests.conftest import FakeMarketDataClient, synthetic_bars


def test_schedules_sum_to_target_and_have_no_negative_orders() -> None:
    bars = synthetic_bars()
    quantity = 50_000
    fake_client = FakeMarketDataClient()
    curve, _, _ = build_historical_volume_curve(fake_client, "NVDA", bars["timestamp_et"].dt.date.iloc[0], "5m", bars)
    schedules = [
        generate_twap_schedule(quantity, bars),
        generate_vwap_schedule(quantity, bars, curve),
        generate_pov_schedule(quantity, bars, 0.10),
        generate_is_schedule(quantity, bars, 0.65),
    ]

    for schedule in schedules:
        assert int(schedule["target_quantity"].sum()) == quantity
        assert int(schedule["target_quantity"].min()) >= 0


def test_is_schedule_front_loads_more_than_twap() -> None:
    bars = synthetic_bars()
    quantity = 50_000
    twap = generate_twap_schedule(quantity, bars)
    is_schedule = generate_is_schedule(quantity, bars, 0.80)
    half = len(bars) // 2

    assert int(is_schedule["target_quantity"].iloc[:half].sum()) > int(
        twap["target_quantity"].iloc[:half].sum()
    )
    assert int(is_schedule["target_quantity"].iloc[:half].sum()) > int(
        is_schedule["target_quantity"].iloc[half:].sum()
    )


def test_pov_respects_target_when_capacity_is_sufficient() -> None:
    bars = synthetic_bars()
    quantity = 25_000
    schedule = generate_pov_schedule(quantity, bars, 0.10)

    assert int(schedule["target_quantity"].sum()) == quantity
    assert (schedule["target_quantity"] <= (bars["volume"] * 0.10 + 1)).all()


def test_strict_pov_reports_unfilled_instead_of_exceeding_cap() -> None:
    bars = synthetic_bars()
    quantity = int(bars["volume"].sum() * 0.20)
    schedule = generate_pov_schedule(quantity, bars, 0.10, pov_mode="strict_cap")

    assert (schedule["target_quantity"] <= schedule["participation_cap_quantity"]).all()
    assert int(schedule["target_quantity"].sum()) < quantity
    assert int(schedule["cap_violation"].sum()) == 0


def test_force_complete_pov_flags_cap_violations() -> None:
    bars = synthetic_bars()
    quantity = int(bars["volume"].sum() * 0.20)
    schedule = generate_pov_schedule(quantity, bars, 0.10, pov_mode="force_complete")

    assert int(schedule["target_quantity"].sum()) == quantity
    assert int(schedule["cap_violation"].sum()) > 0
