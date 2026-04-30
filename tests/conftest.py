from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from execlab.data import EASTERN
from execlab.schemas import BarRecord, MarketDataPayload


def synthetic_bars(
    trade_date: date = date(2026, 4, 20),
    trend_bps: float = 100.0,
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


class FakeMarketDataClient:
    def __init__(self, settings=None, trend_bps: float = 100.0):
        self.settings = settings
        self.trend_bps = trend_bps
        self.calls: list[date] = []

    def fetch_intraday_bars(self, ticker: str, trade_date: date, interval: str = "5m"):
        self.calls.append(trade_date)
        frame = synthetic_bars(trade_date=trade_date, trend_bps=self.trend_bps)
        payload = MarketDataPayload(
            ticker=ticker,
            trade_date=trade_date,
            interval=interval,
            provider="fake-live-provider",
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

    def fetch_daily_history(self, ticker: str, end_date: date, lookback_days: int = 126):
        self.calls.append(end_date)
        dates = pd.bdate_range(end=end_date, periods=lookback_days).date
        seed = sum((idx + 1) * ord(char) for idx, char in enumerate(ticker.upper()))
        rng = np.random.default_rng(seed)
        market_factor = np.linspace(-0.006, 0.007, lookback_days)
        sector_tilt = 1.35 if ticker.upper() in {"NVDA", "XLK", "QQQ"} else 0.85
        noise = rng.normal(0, 0.006, lookback_days)
        if ticker.upper() == "SPY":
            returns = market_factor + rng.normal(0, 0.002, lookback_days)
        elif ticker.upper() in {"XLK", "QQQ"}:
            returns = 1.15 * market_factor + rng.normal(0, 0.003, lookback_days)
        else:
            returns = sector_tilt * market_factor + noise
        close = 100.0 * np.cumprod(1.0 + returns)
        return pd.DataFrame({"date": dates, "close": close})
