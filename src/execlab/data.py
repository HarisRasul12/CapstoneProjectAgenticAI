from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from execlab.config import Settings
from execlab.schemas import BarRecord, MarketDataPayload

EASTERN = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


class MarketDataError(RuntimeError):
    """Raised when live market data cannot be fetched or normalized."""


class MarketDataClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._bar_cache: dict[tuple[str, date, str], tuple[pd.DataFrame, MarketDataPayload]] = {}
        self._daily_cache: dict[tuple[str, date, int], pd.DataFrame] = {}

    def fetch_intraday_bars(self, ticker: str, trade_date: date, interval: str = "5m") -> tuple[pd.DataFrame, MarketDataPayload]:
        cache_key = (ticker.upper(), trade_date, interval)
        if cache_key in self._bar_cache:
            frame, payload = self._bar_cache[cache_key]
            return frame.copy(), payload.model_copy(deep=True)
        if self.settings.data_provider != "yfinance":
            raise MarketDataError(
                f"Unsupported data provider '{self.settings.data_provider}'. "
                "The implemented no-key live provider is yfinance."
            )
        frame, payload = self._fetch_yfinance_intraday(ticker=ticker, trade_date=trade_date, interval=interval)
        self._bar_cache[cache_key] = (frame.copy(), payload.model_copy(deep=True))
        return frame, payload

    def fetch_daily_history(self, ticker: str, end_date: date, lookback_days: int = 126) -> pd.DataFrame:
        cache_key = (ticker.upper(), end_date, lookback_days)
        if cache_key in self._daily_cache:
            return self._daily_cache[cache_key].copy()
        if self.settings.data_provider != "yfinance":
            raise MarketDataError(
                f"Unsupported data provider '{self.settings.data_provider}'. "
                "The implemented no-key live provider is yfinance."
            )
        frame = self._fetch_yfinance_daily(ticker=ticker, end_date=end_date, lookback_days=lookback_days)
        self._daily_cache[cache_key] = frame.copy()
        return frame

    def _fetch_yfinance_intraday(
        self,
        ticker: str,
        trade_date: date,
        interval: str,
    ) -> tuple[pd.DataFrame, MarketDataPayload]:
        warnings: list[str] = []
        self._validate_yfinance_window(trade_date, interval)

        try:
            import yfinance as yf
        except Exception as exc:  # pragma: no cover - depends on environment packaging
            raise MarketDataError("yfinance is not installed; run `uv sync` first.") from exc

        start = trade_date.isoformat()
        end = (trade_date + timedelta(days=1)).isoformat()
        try:
            raw = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                prepost=False,
                progress=False,
                threads=False,
                timeout=self.settings.yfinance_timeout_seconds,
            )
        except Exception as exc:
            raise MarketDataError(f"Live yfinance fetch failed for {ticker} on {trade_date}: {exc}") from exc

        frame = normalize_yfinance_frame(raw, ticker=ticker, trade_date=trade_date)
        if frame.empty:
            raise MarketDataError(
                f"No regular-session intraday bars returned for {ticker} on {trade_date}. "
                "Use a recent U.S. trading day; yfinance intraday history is window-limited."
            )

        if len(frame) < 12:
            warnings.append(
                f"Only {len(frame)} bars were returned; metrics may be noisy for the selected interval/date."
            )

        payload = MarketDataPayload(
            ticker=ticker,
            trade_date=trade_date,
            interval=interval,
            provider="yfinance",
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
            warnings=warnings,
        )
        return frame, payload

    def _fetch_yfinance_daily(self, ticker: str, end_date: date, lookback_days: int) -> pd.DataFrame:
        try:
            import yfinance as yf
        except Exception as exc:  # pragma: no cover - depends on environment packaging
            raise MarketDataError("yfinance is not installed; run `uv sync` first.") from exc

        start = (end_date - timedelta(days=max(lookback_days * 3, lookback_days + 14))).isoformat()
        end = (end_date + timedelta(days=1)).isoformat()
        try:
            raw = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=self.settings.yfinance_timeout_seconds,
            )
        except Exception as exc:
            raise MarketDataError(f"Live yfinance daily fetch failed for {ticker}: {exc}") from exc

        frame = normalize_yfinance_daily_frame(raw, ticker=ticker)
        if frame.empty:
            raise MarketDataError(f"No daily history returned for {ticker}.")
        return frame.tail(lookback_days).reset_index(drop=True)

    @staticmethod
    def _validate_yfinance_window(trade_date: date, interval: str) -> None:
        today_et = datetime.now(EASTERN).date()
        age_days = (today_et - trade_date).days
        if trade_date > today_et:
            raise MarketDataError("Selected trade_date is in the future for U.S. market data.")
        if interval == "1m" and age_days > 30:
            raise MarketDataError(
                "yfinance 1-minute intraday history is limited to a recent window. "
                "Pick a recent date or use 5m/15m."
            )
        if interval in {"5m", "15m"} and age_days > 60:
            raise MarketDataError(
                "yfinance intraday history cannot extend beyond the recent 60-day window. "
                "Pick a recent trading date."
            )


def normalize_yfinance_frame(raw: pd.DataFrame, ticker: str, trade_date: date) -> pd.DataFrame:
    if raw is None or raw.empty:
        return _empty_bars()

    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        # yfinance can return (field, ticker) or (ticker, field); keep the price field level.
        level_values = [list(map(str, frame.columns.get_level_values(level))) for level in range(frame.columns.nlevels)]
        price_fields = {"Open", "High", "Low", "Close", "Volume"}
        selected_level = 0
        for level, values in enumerate(level_values):
            if price_fields.intersection(set(values)):
                selected_level = level
                break
        frame.columns = frame.columns.get_level_values(selected_level)

    rename = {col: str(col).strip().lower().replace(" ", "_") for col in frame.columns}
    frame = frame.rename(columns=rename)
    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise MarketDataError(f"Missing expected yfinance fields for {ticker}: {', '.join(missing)}")

    timestamps = pd.to_datetime(frame.index)
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize(EASTERN)
    else:
        timestamps = timestamps.tz_convert(EASTERN)

    normalized = pd.DataFrame(
        {
            "timestamp_et": timestamps,
            "open": pd.to_numeric(frame["open"], errors="coerce"),
            "high": pd.to_numeric(frame["high"], errors="coerce"),
            "low": pd.to_numeric(frame["low"], errors="coerce"),
            "close": pd.to_numeric(frame["close"], errors="coerce"),
            "volume": pd.to_numeric(frame["volume"], errors="coerce").fillna(0),
        }
    )
    normalized = normalized.dropna(subset=["open", "high", "low", "close"])
    normalized = normalized[normalized["timestamp_et"].dt.date == trade_date]
    normalized = normalized[
        (normalized["timestamp_et"].dt.time >= MARKET_OPEN)
        & (normalized["timestamp_et"].dt.time < MARKET_CLOSE)
    ]
    normalized = normalized.sort_values("timestamp_et").reset_index(drop=True)
    return normalized


def normalize_yfinance_daily_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "close"])

    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        level_values = [list(map(str, frame.columns.get_level_values(level))) for level in range(frame.columns.nlevels)]
        price_fields = {"Close", "Adj Close"}
        selected_level = 0
        for level, values in enumerate(level_values):
            if price_fields.intersection(set(values)):
                selected_level = level
                break
        frame.columns = frame.columns.get_level_values(selected_level)

    rename = {col: str(col).strip().lower().replace(" ", "_") for col in frame.columns}
    frame = frame.rename(columns=rename)
    close_col = "close" if "close" in frame.columns else "adj_close"
    if close_col not in frame.columns:
        raise MarketDataError(f"Missing expected daily close field for {ticker}.")

    dates = pd.to_datetime(frame.index).date
    normalized = pd.DataFrame(
        {
            "date": dates,
            "close": pd.to_numeric(frame[close_col], errors="coerce"),
        }
    )
    normalized = normalized.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    return normalized


def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp_et", "open", "high", "low", "close", "volume"])


def select_window_bars(frame: pd.DataFrame, start_time: time, end_time: time) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    window = frame[
        (frame["timestamp_et"].dt.time >= start_time)
        & (frame["timestamp_et"].dt.time < end_time)
    ].copy()
    return window.reset_index(drop=True)


def validate_market_session(frame: pd.DataFrame, start_time: time, end_time: time) -> list[str]:
    warnings: list[str] = []
    if frame.empty:
        warnings.append("No regular-session bars are available.")
        return warnings
    if frame["volume"].sum() <= 0:
        warnings.append("Returned bars have zero reported volume; VWAP/POV outputs are unreliable.")
    first = frame["timestamp_et"].iloc[0].time()
    last = frame["timestamp_et"].iloc[-1].time()
    if first > start_time:
        warnings.append(f"First available bar is {first}; requested start was {start_time}.")
    if last < end_time and last < time(15, 55):
        warnings.append(f"Last available bar is {last}; requested end was {end_time}.")
    return warnings
