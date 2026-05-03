from __future__ import annotations

import pandas as pd

from execlab.schemas import (
    BetaRiskReport,
    ExecutionRequest,
    MarketEDA,
    PeerStockAnalysisReport,
    PreTradeAnalyticsReport,
    SimulationResult,
    TcaCausalBullet,
    TcaCausalReport,
)


def build_tca_causal_report(
    request: ExecutionRequest,
    eda: MarketEDA,
    simulations: dict[str, SimulationResult],
    schedules: dict[str, pd.DataFrame],
    pretrade: PreTradeAnalyticsReport,
    beta_risk: BetaRiskReport,
    peer_report: PeerStockAnalysisReport | None = None,
) -> TcaCausalReport:
    if not simulations:
        return TcaCausalReport(
            best_algo="Unavailable",
            headline="No simulations were available for cause-effect analysis.",
        )

    ranked = sorted(simulations.values(), key=lambda sim: sim.metrics.arrival_cost_bps)
    best = ranked[0]
    worst = ranked[-1]
    bullets: list[TcaCausalBullet] = []

    price_driver = _price_path_driver(request, eda, schedules)
    if price_driver:
        bullets.append(price_driver)

    bullets.append(
        TcaCausalBullet(
            driver="Benchmark trade-off",
            affected_algos=[best.algo, worst.algo],
            evidence=(
                f"{best.algo} led arrival cost at {best.metrics.arrival_cost_bps:.2f} bps; "
                f"{worst.algo} lagged at {worst.metrics.arrival_cost_bps:.2f} bps. "
                f"Market VWAP was {eda.market_vwap:.4f} versus arrival {eda.arrival_price:.4f}."
            ),
            implication=(
                "The winner is path-dependent: arrival-cost leadership rewards being early on adverse "
                "moves, while VWAP slippage rewards matching the full-day volume curve."
            ),
        )
    )

    pov = simulations.get("POV")
    if pov:
        if pov.metrics.unfilled_quantity > 0:
            implication = (
                "Strict POV protected the participation cap, but the parent order did not finish. "
                "Use force-complete only if finishing matters more than staying under the cap."
            )
        elif pov.metrics.cap_violation_count > 0:
            implication = (
                "Force-complete POV finished the order by exceeding the cap on some bars; treat the "
                "extra shares as explicit urgency/impact risk."
            )
        else:
            implication = "POV respected the selected cap and completed in the available volume."
        bullets.append(
            TcaCausalBullet(
                driver="Participation constraint",
                affected_algos=["POV"],
                evidence=(
                    f"POV requested cap was {request.participation_rate * 100:.2f}%; realized max "
                    f"participation was {pov.metrics.max_participation_rate * 100:.2f}% with "
                    f"{pov.metrics.unfilled_quantity:,} unfilled shares and "
                    f"{pov.metrics.cap_violation_count} cap violations."
                ),
                implication=implication,
            )
        )

    bullets.append(
        TcaCausalBullet(
            driver="Spread and volatility friction",
            affected_algos=[sim.algo for sim in ranked],
            evidence=(
                f"21-day high-low spread proxy was {pretrade.avg_spread_proxy_bps:.2f} bps; "
                f"bar volatility proxy was {pretrade.avg_volatility_bps:.2f} bps. "
                f"Order size was {pretrade.order_size_adv_pct * 100:.2f}% of ADV."
            ),
            implication=(
                "Higher spread and bar volatility make late or high-participation fills more expensive; "
                "lower urgency is only attractive when timing risk is small."
            ),
        )
    )

    bullets.append(
        TcaCausalBullet(
            driver="Systematic versus idiosyncratic timing risk",
            affected_algos=[best.algo],
            evidence=(
                f"{request.ticker} maps to {beta_risk.sector_etf} ({beta_risk.sector_label}); "
                f"market beta {beta_risk.beta_market:.2f}, sector beta {beta_risk.beta_sector:.2f}, "
                f"idiosyncratic risk {beta_risk.idiosyncratic_risk_bps:.2f} bps over the window."
            ),
            implication=(
                "If systematic risk dominates, compare execution timing against SPY/sector moves; "
                "if idiosyncratic risk dominates, the stock-specific tape matters more than the market."
            ),
        )
    )

    if peer_report and peer_report.analyzed_count > 0:
        bullets.append(
            TcaCausalBullet(
                driver="Peer crowding and impact pressure",
                affected_algos=[best.algo],
                evidence=(
                    f"{peer_report.analyzed_count} peers were analyzed in {peer_report.sector_etf}; "
                    f"average correlation {peer_report.average_peer_correlation:.2f}, crowding score "
                    f"{peer_report.crowding_score:.2f}, median peer move "
                    f"{peer_report.median_peer_move_bps:.2f} bps."
                ),
                implication=(
                    f"PeerClusterAgent recommends {peer_report.urgency_recommendation}: "
                    f"{peer_report.rationale}"
                ),
            )
        )

    if request.limit_price is not None:
        total_unfilled = sum(sim.metrics.unfilled_quantity for sim in simulations.values())
        bullets.append(
            TcaCausalBullet(
                driver="Limit feasibility",
                affected_algos=[sim.algo for sim in ranked if sim.metrics.unfilled_quantity > 0],
                evidence=f"Limit {request.limit_price:.4f} left {total_unfilled:,} total unfilled shares across algos.",
                implication="The limit controls price but can turn benchmark comparison into a completion problem.",
            )
        )

    headline = (
        f"{best.algo} is best by arrival cost because its fill timing best matched the realized "
        f"price path and liquidity constraints; {worst.algo} paid the weakest path/benchmark trade-off."
    )
    return TcaCausalReport(
        best_algo=best.algo,
        headline=headline,
        bullets=bullets,
        caveats=[
            "Cause-effect bullets are derived from bar data and schedule diagnostics, not venue-level order-book data.",
            "Positive bps are worse for the selected side; negative bps are favorable versus the benchmark.",
        ],
    )


def _price_path_driver(
    request: ExecutionRequest,
    eda: MarketEDA,
    schedules: dict[str, pd.DataFrame],
) -> TcaCausalBullet | None:
    if not schedules:
        return None
    weighted_times = {
        algo: _weighted_time_fraction(schedule)
        for algo, schedule in schedules.items()
        if not schedule.empty and float(schedule["target_quantity"].sum()) > 0
    }
    if not weighted_times:
        return None
    earliest = min(weighted_times, key=weighted_times.get)
    latest = max(weighted_times, key=weighted_times.get)
    rising = eda.price_move_bps > 0
    adverse_for_buy = request.side == "buy" and rising
    adverse_for_sell = request.side == "sell" and not rising
    if adverse_for_buy or adverse_for_sell:
        implication = (
            f"The tape moved against a {request.side}; earlier schedules such as {earliest} reduce "
            f"exposure to later adverse prices."
        )
    else:
        implication = (
            f"The tape moved favorably for a {request.side}; slower schedules such as {latest} can look "
            "better versus arrival, but that is realized drift rather than guaranteed skill."
        )
    return TcaCausalBullet(
        driver="Price-path timing",
        affected_algos=[earliest, latest],
        evidence=(
            f"Window price move was {eda.price_move_bps:.2f} bps. "
            f"{earliest} weighted fill time was {weighted_times[earliest] * 100:.1f}% through the window; "
            f"{latest} was {weighted_times[latest] * 100:.1f}%."
        ),
        implication=implication,
    )


def _weighted_time_fraction(schedule: pd.DataFrame) -> float:
    frame = schedule.reset_index(drop=True)
    total_qty = float(frame["target_quantity"].sum())
    if total_qty <= 0 or len(frame) <= 1:
        return 0.0
    positions = pd.Series(range(len(frame)), dtype=float) / (len(frame) - 1)
    return float((positions * frame["target_quantity"].astype(float)).sum() / total_qty)
