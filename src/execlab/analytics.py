from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from execlab.data import MarketDataClient, MarketDataError
from execlab.schemas import MarketEDA


def bar_vwap_proxy(frame: pd.DataFrame) -> pd.Series:
    return (frame["open"] + frame["high"] + frame["low"] + frame["close"]) / 4.0


def market_vwap(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    prices = bar_vwap_proxy(frame)
    volume = frame["volume"].clip(lower=0)
    if float(volume.sum()) <= 0:
        return float(prices.mean())
    return float(np.average(prices, weights=volume))


def build_market_eda(
    ticker: str,
    trade_date: date,
    interval: str,
    all_bars: pd.DataFrame,
    window_bars: pd.DataFrame,
    volume_curve_source: str,
) -> MarketEDA:
    if all_bars.empty or window_bars.empty:
        raise ValueError("Cannot build market EDA without intraday bars.")

    arrival = float(window_bars["open"].iloc[0])
    close = float(window_bars["close"].iloc[-1])
    returns = window_bars["close"].pct_change().dropna()
    realized_vol_bps = float(returns.std(ddof=0) * 10_000) if not returns.empty else 0.0
    mids = ((window_bars["high"] + window_bars["low"]) / 2.0).replace(0, np.nan)
    spread_proxy = ((window_bars["high"] - window_bars["low"]) / mids * 10_000).replace([np.inf, -np.inf], np.nan)
    spread_proxy_bps = float(spread_proxy.dropna().mean()) if not spread_proxy.dropna().empty else 0.0

    return MarketEDA(
        ticker=ticker,
        trade_date=trade_date,
        interval=interval,
        bar_count=int(len(window_bars)),
        arrival_price=arrival,
        close_price=close,
        market_vwap=market_vwap(window_bars),
        total_volume=float(all_bars["volume"].sum()),
        window_volume=float(window_bars["volume"].sum()),
        price_move_bps=float((close - arrival) / arrival * 10_000) if arrival else 0.0,
        realized_volatility_bps=realized_vol_bps,
        high_low_spread_proxy_bps=spread_proxy_bps,
        volume_curve_source=volume_curve_source,
    )


def build_historical_volume_curve(
    client: MarketDataClient,
    ticker: str,
    trade_date: date,
    interval: str,
    current_window_bars: pd.DataFrame,
    lookback_days: int = 5,
) -> tuple[pd.Series, str, list[str]]:
    """Return expected volume weights aligned to current window bars.

    The function fetches recent prior trading days live. If it cannot collect enough
    aligned bars, it returns the same-day curve and a clear source label.
    """

    warnings: list[str] = []
    if current_window_bars.empty:
        return pd.Series(dtype=float), "unavailable", ["No current bars for curve alignment."]

    current_times = current_window_bars["timestamp_et"].dt.strftime("%H:%M")
    current_volumes = current_window_bars["volume"].astype(float).clip(lower=0)
    fallback = _weights_from_volume(current_volumes, index=current_window_bars.index)

    prior_curves: list[pd.Series] = []
    cursor = trade_date - timedelta(days=1)
    attempts = 0
    while len(prior_curves) < lookback_days and attempts < lookback_days * 4:
        attempts += 1
        if cursor.weekday() >= 5:
            cursor -= timedelta(days=1)
            continue
        try:
            prior_frame, _ = client.fetch_intraday_bars(ticker, cursor, interval)
        except MarketDataError as exc:
            warnings.append(f"Prior curve fetch skipped for {cursor}: {exc}")
            cursor -= timedelta(days=1)
            continue
        prior = prior_frame.copy()
        prior["time_key"] = prior["timestamp_et"].dt.strftime("%H:%M")
        aligned = prior.set_index("time_key")["volume"].reindex(current_times.tolist()).fillna(0)
        if float(aligned.sum()) > 0 and aligned.gt(0).sum() >= max(5, len(current_times) // 4):
            prior_curves.append(_weights_from_volume(aligned, index=current_window_bars.index))
        cursor -= timedelta(days=1)

    if len(prior_curves) >= 2:
        curve = pd.concat(prior_curves, axis=1).mean(axis=1)
        curve = curve / float(curve.sum()) if float(curve.sum()) > 0 else fallback
        return curve, f"historical_{len(prior_curves)}_day_live_curve", warnings

    warnings.append(
        "Insufficient prior intraday bars for a stable historical curve; using same-day volume curve."
    )
    return fallback, "same_day_volume_curve_fallback", warnings


def _weights_from_volume(values: pd.Series, index: pd.Index) -> pd.Series:
    volumes = pd.to_numeric(values, errors="coerce").fillna(0).clip(lower=0)
    if float(volumes.sum()) <= 0:
        return pd.Series(np.ones(len(index)) / max(1, len(index)), index=index)
    weights = volumes / float(volumes.sum())
    weights.index = index
    return weights.astype(float)

