from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from execlab.analytics import bar_vwap_proxy
from execlab.data import MarketDataClient, MarketDataError
from execlab.schemas import (
    CostBreakdownItem,
    ExpectedCostModelReport,
    ExecutionRequest,
    PreTradeAnalyticsReport,
    PreTradeCurvePoint,
    RegressionCoefficient,
    ThroughDayCostPoint,
)


def fetch_lookback_sessions(
    client: MarketDataClient,
    ticker: str,
    anchor_date: date,
    interval: str,
    requested_sessions: int = 21,
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    frames: list[pd.DataFrame] = []
    cursor = anchor_date - timedelta(days=1)
    attempts = 0
    while len(frames) < requested_sessions and attempts < requested_sessions * 4:
        attempts += 1
        if cursor.weekday() >= 5:
            cursor -= timedelta(days=1)
            continue
        try:
            frame, _ = client.fetch_intraday_bars(ticker=ticker, trade_date=cursor, interval=interval)
        except MarketDataError as exc:
            warnings.append(f"21-day pre-trade fetch skipped for {cursor}: {exc}")
            cursor -= timedelta(days=1)
            continue
        if not frame.empty:
            enriched = frame.copy()
            enriched["session_date"] = cursor
            enriched["time_key"] = enriched["timestamp_et"].dt.strftime("%H:%M")
            frames.append(enriched)
        cursor -= timedelta(days=1)

    if not frames:
        return pd.DataFrame(), warnings
    return pd.concat(frames, ignore_index=True).sort_values(["session_date", "timestamp_et"]), warnings


def build_pretrade_analytics(
    request: ExecutionRequest,
    current_window_bars: pd.DataFrame,
    lookback_bars: pd.DataFrame,
    requested_sessions: int = 21,
    warnings: list[str] | None = None,
) -> PreTradeAnalyticsReport:
    warnings = list(warnings or [])
    if lookback_bars.empty:
        warnings.append("No lookback bars were available; pre-trade analytics are degraded.")
        return PreTradeAnalyticsReport(
            lookback_sessions=0,
            requested_sessions=requested_sessions,
            adv_shares=0.0,
            order_size_adv_pct=0.0,
            avg_spread_proxy_bps=0.0,
            avg_volatility_bps=0.0,
            time_risk_score=0.0,
            current_vs_21d_volume=0.0,
            warnings=warnings,
        )

    hist = _enrich_intraday_features(lookback_bars)
    sessions = hist["session_date"].nunique()
    adv = float(hist.groupby("session_date")["volume"].sum().mean())
    order_size_adv_pct = float(request.quantity / adv) if adv > 0 else 0.0
    avg_spread = float(hist["spread_proxy_bps"].replace([np.inf, -np.inf], np.nan).dropna().mean())
    avg_vol = float(hist["abs_return_bps"].replace([np.inf, -np.inf], np.nan).dropna().mean())

    curve_frame = (
        hist.groupby("time_key")
        .agg(
            avg_volume=("volume", "mean"),
            avg_spread_proxy_bps=("spread_proxy_bps", "mean"),
            avg_volatility_bps=("abs_return_bps", "mean"),
        )
        .reset_index()
        .sort_values("time_key")
    )
    total_curve_volume = float(curve_frame["avg_volume"].sum())
    if total_curve_volume > 0:
        curve_frame["avg_volume_pct"] = curve_frame["avg_volume"] / total_curve_volume
    else:
        curve_frame["avg_volume_pct"] = 0.0
    curve_frame["time_risk_score"] = _time_risk_scores(curve_frame["time_key"].tolist())

    current = current_window_bars.copy()
    current_volume = float(current["volume"].sum()) if not current.empty else 0.0
    current_times = set(current["timestamp_et"].dt.strftime("%H:%M").tolist()) if not current.empty else set()
    hist_window_volume = float(curve_frame[curve_frame["time_key"].isin(current_times)]["avg_volume"].sum())
    current_vs_21d = current_volume / hist_window_volume if hist_window_volume > 0 else 0.0
    time_risk = float(curve_frame["time_risk_score"].mean()) if not curve_frame.empty else 0.0

    points = [
        PreTradeCurvePoint(
            time_key=str(row.time_key),
            avg_volume=float(row.avg_volume),
            avg_volume_pct=float(row.avg_volume_pct),
            avg_spread_proxy_bps=float(row.avg_spread_proxy_bps),
            avg_volatility_bps=float(row.avg_volatility_bps),
            time_risk_score=float(row.time_risk_score),
        )
        for row in curve_frame.itertuples(index=False)
    ]

    return PreTradeAnalyticsReport(
        lookback_sessions=int(sessions),
        requested_sessions=requested_sessions,
        adv_shares=adv,
        order_size_adv_pct=order_size_adv_pct,
        avg_spread_proxy_bps=avg_spread,
        avg_volatility_bps=avg_vol,
        time_risk_score=time_risk,
        current_vs_21d_volume=float(current_vs_21d),
        curve=points,
        warnings=warnings,
    )


def fit_expected_cost_model(
    request: ExecutionRequest,
    pretrade: PreTradeAnalyticsReport,
    lookback_bars: pd.DataFrame,
    seed: int,
) -> ExpectedCostModelReport:
    if lookback_bars.empty or not pretrade.curve:
        return ExpectedCostModelReport(
            expected_cost_bps=0.0,
            model_r2=0.0,
            observation_count=0,
            caveats=["Expected-cost regression unavailable because 21-day lookback bars were unavailable."],
        )

    hist = _enrich_intraday_features(lookback_bars)
    hist["future_price"] = hist.groupby("session_date")["bar_price"].shift(-6)
    hist = hist.dropna(subset=["future_price"]).copy()
    hist["signed_future_move_bps"] = np.where(
        request.side == "buy",
        (hist["future_price"] - hist["bar_price"]) / hist["bar_price"] * 10_000,
        (hist["bar_price"] - hist["future_price"]) / hist["bar_price"] * 10_000,
    )
    adv = max(pretrade.adv_shares, 1.0)
    hist["participation_rate"] = np.minimum(0.50, request.quantity / np.maximum(hist["volume"], 1.0))
    hist["order_size_adv_pct"] = request.quantity / adv
    hist["relative_volume"] = hist["volume"] / max(float(hist["volume"].mean()), 1.0)
    hist["time_risk"] = _time_risk_scores(hist["time_key"].tolist()).to_numpy()
    hist["drift_proxy_bps"] = hist.groupby("session_date")["bar_price"].pct_change().fillna(0) * 10_000
    hist["target_cost_bps"] = (
        0.5 * hist["spread_proxy_bps"]
        + 10.0 * hist["participation_rate"]
        + 0.12 * hist["abs_return_bps"] * hist["time_risk"]
        + hist["signed_future_move_bps"]
    )

    feature_names = [
        "intercept",
        "participation_rate",
        "spread_proxy_bps",
        "abs_return_bps",
        "time_risk",
        "relative_volume",
        "order_size_adv_pct",
        "drift_proxy_bps",
    ]
    x = np.column_stack(
        [
            np.ones(len(hist)),
            hist["participation_rate"].to_numpy(float),
            hist["spread_proxy_bps"].fillna(0).to_numpy(float),
            hist["abs_return_bps"].fillna(0).to_numpy(float),
            hist["time_risk"].fillna(0).to_numpy(float),
            hist["relative_volume"].fillna(1).to_numpy(float),
            hist["order_size_adv_pct"].fillna(0).to_numpy(float),
            hist["drift_proxy_bps"].fillna(0).to_numpy(float),
        ]
    )
    y = hist["target_cost_bps"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    predicted = x @ beta
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    duration_fraction = _duration_fraction(request)
    current_features = np.array(
        [
            1.0,
            float(request.participation_rate),
            pretrade.avg_spread_proxy_bps,
            pretrade.avg_volatility_bps,
            pretrade.time_risk_score,
            max(pretrade.current_vs_21d_volume, 0.1),
            pretrade.order_size_adv_pct,
            request.drift_bps_per_day * duration_fraction,
        ],
        dtype=float,
    )
    regression_expected = float(current_features @ beta)
    spread_cost = pretrade.avg_spread_proxy_bps / 2.0
    impact_cost = request.impact_bps_per_10pct * (request.participation_rate / 0.10) * max(1.0, pretrade.order_size_adv_pct * 10)
    timing_risk = pretrade.avg_volatility_bps * np.sqrt(max(duration_fraction, 0.01)) * 0.35
    drift_risk = request.drift_bps_per_day * duration_fraction * (1 if request.side == "buy" else -1)
    limit_risk = _limit_risk_bps(request, pretrade)
    blended_expected = float(0.45 * regression_expected + 0.55 * (spread_cost + impact_cost + timing_risk + drift_risk + limit_risk))

    through_day = _through_day_cost_points(
        pretrade=pretrade,
        request=request,
        seed=seed,
        spread_cost=spread_cost,
        impact_cost=impact_cost,
    )

    return ExpectedCostModelReport(
        expected_cost_bps=blended_expected,
        model_r2=float(max(0.0, min(1.0, r2))),
        observation_count=int(len(hist)),
        coefficients=[
            RegressionCoefficient(feature=name, coefficient=float(value))
            for name, value in zip(feature_names, beta, strict=True)
        ],
        cost_breakdown=[
            CostBreakdownItem(
                component="Spread proxy",
                bps=float(spread_cost),
                description="Half of the 21-day average high-low spread proxy.",
            ),
            CostBreakdownItem(
                component="Impact",
                bps=float(impact_cost),
                description="Participation-driven impact using the configured bps-per-10% input.",
            ),
            CostBreakdownItem(
                component="Timing risk",
                bps=float(timing_risk),
                description="Volatility scaled by remaining execution horizon.",
            ),
            CostBreakdownItem(
                component="Drift",
                bps=float(drift_risk),
                description="User-specified directional drift over the execution window.",
            ),
            CostBreakdownItem(
                component="Limit/unfilled risk",
                bps=float(limit_risk),
                description="Penalty proxy when a limit price may prevent completion.",
            ),
        ],
        through_day_cost=through_day,
        caveats=[
            "This is an educational expected-cost model fit on public OHLCV bars.",
            "Spread uses a high-low proxy, not true NBBO/bid-ask data.",
            "Regression coefficients are transparent diagnostics, not production TCA estimates.",
        ],
    )


def _enrich_intraday_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    if "time_key" not in enriched.columns:
        enriched["time_key"] = enriched["timestamp_et"].dt.strftime("%H:%M")
    if "session_date" not in enriched.columns:
        enriched["session_date"] = enriched["timestamp_et"].dt.date
    enriched["bar_price"] = bar_vwap_proxy(enriched)
    mid = ((enriched["high"] + enriched["low"]) / 2.0).replace(0, np.nan)
    enriched["spread_proxy_bps"] = ((enriched["high"] - enriched["low"]) / mid * 10_000).replace([np.inf, -np.inf], np.nan).fillna(0)
    enriched["return_bps"] = enriched.groupby("session_date")["close"].pct_change().fillna(0) * 10_000
    enriched["abs_return_bps"] = enriched["return_bps"].abs()
    return enriched


def _time_risk_scores(time_keys: list[str]) -> pd.Series:
    scores = []
    for key in time_keys:
        hour, minute = [int(part) for part in key.split(":")]
        minutes = hour * 60 + minute
        close_minutes = 16 * 60
        open_minutes = 9 * 60 + 30
        remaining = max(0, close_minutes - minutes)
        total = close_minutes - open_minutes
        scores.append(float(np.sqrt(remaining / total)) if total > 0 else 0.0)
    return pd.Series(scores)


def _duration_fraction(request: ExecutionRequest) -> float:
    start = request.start_time.hour * 60 + request.start_time.minute
    end = request.end_time.hour * 60 + request.end_time.minute
    return max(1, end - start) / 390.0


def _limit_risk_bps(request: ExecutionRequest, pretrade: PreTradeAnalyticsReport) -> float:
    if request.limit_price is None:
        return 0.0
    return min(25.0, 2.5 + 0.2 * pretrade.avg_volatility_bps + 15.0 * pretrade.order_size_adv_pct)


def _through_day_cost_points(
    pretrade: PreTradeAnalyticsReport,
    request: ExecutionRequest,
    seed: int,
    spread_cost: float,
    impact_cost: float,
) -> list[ThroughDayCostPoint]:
    rng = np.random.default_rng(seed)
    points: list[ThroughDayCostPoint] = []
    for point in pretrade.curve:
        hour, minute = [int(part) for part in point.time_key.split(":")]
        minutes = hour * 60 + minute
        remaining_fraction = max(0.01, (16 * 60 - minutes) / 390.0)
        timing_sigma = max(point.avg_volatility_bps, pretrade.avg_volatility_bps, 0.5) * np.sqrt(remaining_fraction)
        drift = request.drift_bps_per_day * remaining_fraction * (1 if request.side == "buy" else -1)
        draws = spread_cost + impact_cost + drift + rng.normal(0, timing_sigma, size=300)
        points.append(
            ThroughDayCostPoint(
                time_key=point.time_key,
                p10_bps=float(np.percentile(draws, 10)),
                p50_bps=float(np.percentile(draws, 50)),
                p90_bps=float(np.percentile(draws, 90)),
            )
        )
    return points
