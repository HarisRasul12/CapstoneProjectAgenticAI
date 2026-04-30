from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from execlab.data import MarketDataClient, MarketDataError
from execlab.data import select_window_bars
from execlab.schemas import (
    BetaRiskReport,
    ExecutionRequest,
    IndexComparisonPoint,
    PeerStockAnalysisReport,
    PeerStockMetric,
    RegressionCoefficient,
)


MARKET_ETF = "SPY"

SECTOR_ETF_MAP: dict[str, tuple[str, str, set[str]]] = {
    "XLK": (
        "Technology",
        "mega-cap software, semiconductors, and hardware",
        {
            "AAPL",
            "MSFT",
            "NVDA",
            "AVGO",
            "AMD",
            "CRM",
            "ORCL",
            "ADBE",
            "CSCO",
            "QCOM",
            "TXN",
            "INTC",
            "IBM",
            "NOW",
            "PANW",
            "MU",
            "SNOW",
            "PLTR",
        },
    ),
    "XLC": (
        "Communication services",
        "internet media, telecom, and streaming",
        {"GOOGL", "GOOG", "META", "NFLX", "DIS", "CMCSA", "TMUS", "VZ", "T", "CHTR"},
    ),
    "XLY": (
        "Consumer discretionary",
        "consumer cyclicals, e-commerce, autos, and retail",
        {"AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "BKNG", "TJX", "CMG"},
    ),
    "XLP": (
        "Consumer staples",
        "defensive staples and household products",
        {"WMT", "COST", "PG", "KO", "PEP", "PM", "MO", "MDLZ", "CL", "KMB"},
    ),
    "XLF": (
        "Financials",
        "banks, brokers, cards, and insurers",
        {"JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "BRK-B", "BRK.A"},
    ),
    "XLV": (
        "Health care",
        "pharma, managed care, medical devices, and biotech",
        {"LLY", "UNH", "JNJ", "MRK", "ABBV", "PFE", "TMO", "ABT", "DHR", "BMY", "AMGN"},
    ),
    "XLE": (
        "Energy",
        "integrated oil, E&P, oil services, and pipelines",
        {"XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "OXY", "VLO", "HAL"},
    ),
    "XLI": (
        "Industrials",
        "capital goods, aerospace, transport, and industrial services",
        {"GE", "CAT", "RTX", "HON", "UNP", "UPS", "DE", "BA", "LMT", "ETN", "MMM"},
    ),
    "XLB": (
        "Materials",
        "chemicals, metals, mining, and packaging",
        {"LIN", "APD", "SHW", "ECL", "FCX", "NEM", "DOW", "DD", "NUE", "MLM"},
    ),
    "XLU": (
        "Utilities",
        "regulated electric, gas, and water utilities",
        {"NEE", "SO", "DUK", "AEP", "SRE", "D", "EXC", "XEL", "PEG", "ED"},
    ),
    "XLRE": (
        "Real estate",
        "REITs and listed real estate operating companies",
        {"PLD", "AMT", "EQIX", "WELL", "SPG", "PSA", "O", "DLR", "CCI", "VICI"},
    ),
}

KNOWN_ETFS = {
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "XLK",
    "XLC",
    "XLY",
    "XLP",
    "XLF",
    "XLV",
    "XLE",
    "XLI",
    "XLB",
    "XLU",
    "XLRE",
}


@dataclass(frozen=True)
class EtfMapping:
    sector_etf: str
    sector_label: str
    confidence: float
    reason: str


def map_ticker_to_etf(ticker: str) -> EtfMapping:
    symbol = ticker.upper().strip()
    if symbol in KNOWN_ETFS:
        return EtfMapping(
            sector_etf=symbol if symbol != "SPY" else "SPY",
            sector_label="ETF / basket",
            confidence=0.90,
            reason=f"{symbol} is already an ETF/basket, so it is used as its own comparison ETF.",
        )
    for etf, (label, description, members) in SECTOR_ETF_MAP.items():
        if symbol in members:
            return EtfMapping(
                sector_etf=etf,
                sector_label=label,
                confidence=0.85,
                reason=f"{symbol} maps to {etf} because it is a representative {description} name.",
            )
    return EtfMapping(
        sector_etf="SPY",
        sector_label="Broad market",
        confidence=0.35,
        reason=f"No high-confidence sector mapping was available for {symbol}; SPY is used as fallback.",
    )


def build_beta_risk_report(
    request: ExecutionRequest,
    client: MarketDataClient,
    lookback_days: int = 126,
) -> BetaRiskReport:
    mapping = map_ticker_to_etf(request.ticker)
    warnings: list[str] = []
    try:
        ticker_prices = client.fetch_daily_history(request.ticker, request.trade_date, lookback_days)
        market_prices = client.fetch_daily_history(MARKET_ETF, request.trade_date, lookback_days)
        sector_prices = client.fetch_daily_history(mapping.sector_etf, request.trade_date, lookback_days)
    except MarketDataError as exc:
        warnings.append(f"Beta risk model unavailable: {exc}")
        return _empty_report(request, mapping, lookback_days, warnings)

    returns = _aligned_returns(
        request.ticker,
        ticker_prices,
        MARKET_ETF,
        market_prices,
        mapping.sector_etf,
        sector_prices,
    )
    if len(returns) < 20:
        warnings.append("Beta risk model degraded: fewer than 20 aligned daily return observations.")
        return _empty_report(request, mapping, lookback_days, warnings)

    y = returns["asset"].to_numpy(float)
    market = returns["market"].to_numpy(float)
    sector = returns["sector"].to_numpy(float)
    use_sector = mapping.sector_etf != MARKET_ETF and not np.allclose(market, sector)

    if use_sector:
        x = np.column_stack([np.ones(len(returns)), market, sector])
        names = ["intercept", f"beta_{MARKET_ETF}", f"beta_{mapping.sector_etf}"]
    else:
        x = np.column_stack([np.ones(len(returns)), market])
        names = ["intercept", f"beta_{MARKET_ETF}"]

    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    predicted = x @ coefficients
    residual = y - predicted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    beta_market = float(coefficients[1]) if len(coefficients) > 1 else 0.0
    beta_sector = float(coefficients[2]) if use_sector and len(coefficients) > 2 else 0.0
    duration_fraction = _duration_fraction(request)
    side_sign = 1.0 if request.side == "buy" else -1.0

    ticker_daily_vol_bps = _vol_bps(y)
    market_daily_vol_bps = _vol_bps(market)
    sector_daily_vol_bps = _vol_bps(sector)
    systematic_risk = abs(beta_market) * market_daily_vol_bps * np.sqrt(duration_fraction)
    sector_risk = abs(beta_sector) * sector_daily_vol_bps * np.sqrt(duration_fraction)
    idio_risk = _vol_bps(residual) * np.sqrt(duration_fraction)
    total_risk = float(np.sqrt(systematic_risk**2 + sector_risk**2 + idio_risk**2))

    index_comparison, intraday_warnings = build_intraday_index_comparison(
        request=request,
        client=client,
        sector_etf=mapping.sector_etf,
    )
    warnings.extend(intraday_warnings)

    return BetaRiskReport(
        ticker=request.ticker,
        market_etf=MARKET_ETF,
        sector_etf=mapping.sector_etf,
        sector_label=mapping.sector_label,
        mapping_confidence=mapping.confidence,
        mapping_reason=mapping.reason,
        lookback_days=lookback_days,
        observation_count=int(len(returns)),
        beta_market=beta_market,
        beta_sector=beta_sector,
        correlation_market=_correlation(y, market),
        correlation_sector=_correlation(y, sector),
        r_squared=float(max(0.0, min(1.0, r_squared))),
        ticker_daily_vol_bps=ticker_daily_vol_bps,
        market_daily_vol_bps=market_daily_vol_bps,
        sector_daily_vol_bps=sector_daily_vol_bps,
        systematic_risk_bps=float(systematic_risk),
        sector_risk_bps=float(sector_risk),
        idiosyncratic_risk_bps=float(idio_risk),
        total_timing_risk_bps=total_risk,
        systematic_cost_bps=float(side_sign * beta_market * np.mean(market) * 10_000 * duration_fraction),
        idiosyncratic_cost_bps=float(side_sign * np.mean(residual) * 10_000 * duration_fraction),
        coefficients=[
            RegressionCoefficient(feature=name, coefficient=float(value), units="daily return beta")
            for name, value in zip(names, coefficients, strict=True)
        ],
        index_comparison=index_comparison,
        warnings=warnings,
    )


def build_intraday_index_comparison(
    request: ExecutionRequest,
    client: MarketDataClient,
    sector_etf: str,
) -> tuple[list[IndexComparisonPoint], list[str]]:
    warnings: list[str] = []
    try:
        ticker_bars, _ = client.fetch_intraday_bars(request.ticker, request.trade_date, request.interval)
        market_bars, _ = client.fetch_intraday_bars(MARKET_ETF, request.trade_date, request.interval)
        sector_bars, _ = client.fetch_intraday_bars(sector_etf, request.trade_date, request.interval)
    except MarketDataError as exc:
        return [], [f"Index comparison unavailable: {exc}"]

    frames = []
    for alias, frame in [
        ("ticker_return_bps", select_window_bars(ticker_bars, request.start_time, request.end_time)),
        ("market_return_bps", select_window_bars(market_bars, request.start_time, request.end_time)),
        ("sector_return_bps", select_window_bars(sector_bars, request.start_time, request.end_time)),
    ]:
        if frame.empty:
            warnings.append(f"Index comparison missing {alias.replace('_return_bps', '')} bars.")
            continue
        temp = frame[["timestamp_et", "close"]].copy()
        temp["time_key"] = temp["timestamp_et"].dt.strftime("%H:%M")
        base = float(temp["close"].iloc[0])
        temp[alias] = (temp["close"] / base - 1.0) * 10_000 if base > 0 else 0.0
        frames.append(temp[["time_key", alias]])

    if len(frames) < 3:
        return [], warnings
    merged = frames[0].merge(frames[1], on="time_key", how="inner").merge(frames[2], on="time_key", how="inner")
    points = [
        IndexComparisonPoint(
            time_key=str(row.time_key),
            ticker_return_bps=float(row.ticker_return_bps),
            market_return_bps=float(row.market_return_bps),
            sector_return_bps=float(row.sector_return_bps),
        )
        for row in merged.itertuples(index=False)
    ]
    return points, warnings


def build_peer_stock_analysis(
    request: ExecutionRequest,
    client: MarketDataClient,
    beta_risk: BetaRiskReport,
    lookback_days: int = 126,
    max_candidates: int = 10,
) -> PeerStockAnalysisReport:
    mapping = map_ticker_to_etf(request.ticker)
    members = set()
    if mapping.sector_etf in SECTOR_ETF_MAP:
        members = set(SECTOR_ETF_MAP[mapping.sector_etf][2])
    if not members and request.ticker in KNOWN_ETFS:
        members = KNOWN_ETFS - {request.ticker}
    candidates = sorted(symbol for symbol in members if symbol != request.ticker)[:max_candidates]
    warnings: list[str] = []
    if not candidates:
        return _empty_peer_report(
            beta_risk=beta_risk,
            candidate_count=0,
            warnings=[f"No peer basket is configured for {request.ticker}; peer analysis unavailable."],
        )

    try:
        target_prices = client.fetch_daily_history(request.ticker, request.trade_date, lookback_days)
    except MarketDataError as exc:
        return _empty_peer_report(
            beta_risk=beta_risk,
            candidate_count=len(candidates),
            warnings=[f"Peer analysis unavailable for target: {exc}"],
        )

    target_returns = _single_return_frame("target", target_prices)
    if len(target_returns) < 20:
        return _empty_peer_report(
            beta_risk=beta_risk,
            candidate_count=len(candidates),
            warnings=["Peer analysis degraded: target has fewer than 20 daily return observations."],
        )
    target_recent_move = _recent_move_bps(target_prices)

    peers: list[PeerStockMetric] = []
    for peer in candidates:
        try:
            peer_prices = client.fetch_daily_history(peer, request.trade_date, lookback_days)
        except MarketDataError as exc:
            warnings.append(f"Peer {peer} skipped: {exc}")
            continue
        peer_returns = _single_return_frame("peer", peer_prices)
        merged = target_returns.merge(peer_returns, on="date", how="inner")
        if len(merged) < 20:
            warnings.append(f"Peer {peer} skipped: insufficient aligned returns.")
            continue
        target = merged["target"].to_numpy(float)
        peer_values = merged["peer"].to_numpy(float)
        corr = _correlation(target, peer_values)
        variance = float(np.var(target, ddof=1)) if len(target) > 1 else 0.0
        beta_to_target = float(np.cov(peer_values, target, ddof=1)[0, 1] / variance) if variance > 0 else 0.0
        peer_move = _recent_move_bps(peer_prices)
        move_gap = peer_move - target_recent_move
        cluster = _peer_cluster(corr, peer_move, target_recent_move)
        peers.append(
            PeerStockMetric(
                ticker=peer,
                correlation=corr,
                beta_to_target=beta_to_target,
                recent_move_bps=peer_move,
                target_move_bps=target_recent_move,
                move_gap_bps=float(move_gap),
                cluster=cluster,
                impact_signal=_impact_signal(corr, peer_move, request.side),
            )
        )

    peers = sorted(peers, key=lambda item: (abs(item.correlation), -abs(item.move_gap_bps)), reverse=True)[:6]
    if not peers:
        return _empty_peer_report(
            beta_risk=beta_risk,
            candidate_count=len(candidates),
            warnings=warnings or ["No peers had enough aligned history."],
        )

    avg_corr = float(np.mean([max(0.0, peer.correlation) for peer in peers]))
    median_move = float(np.median([peer.recent_move_bps for peer in peers]))
    same_direction_count = sum(np.sign(peer.recent_move_bps) == np.sign(target_recent_move) for peer in peers)
    crowding = float(min(1.0, avg_corr * (0.55 + 0.45 * same_direction_count / max(1, len(peers)))))
    recommendation, rationale = _peer_urgency_recommendation(
        request=request,
        crowding_score=crowding,
        median_peer_move_bps=median_move,
        target_move_bps=target_recent_move,
        beta_risk=beta_risk,
    )
    return PeerStockAnalysisReport(
        sector_etf=mapping.sector_etf,
        candidate_count=len(candidates),
        analyzed_count=len(peers),
        average_peer_correlation=avg_corr,
        crowding_score=crowding,
        median_peer_move_bps=median_move,
        target_recent_move_bps=target_recent_move,
        urgency_recommendation=recommendation,
        rationale=rationale,
        market_impact_note=(
            "High peer correlation and same-direction peer moves suggest crowded sector flow; "
            "that can raise short-horizon impact and timing risk. Divergent peers suggest the move is "
            "more stock-specific, so schedule choice can lean more on spread and liquidity."
        ),
        peers=peers,
        warnings=warnings,
    )


def _aligned_returns(
    ticker: str,
    ticker_prices: pd.DataFrame,
    market_etf: str,
    market_prices: pd.DataFrame,
    sector_etf: str,
    sector_prices: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for alias, frame in [
        ("asset", ticker_prices),
        ("market", market_prices),
        ("sector", sector_prices),
    ]:
        ret = frame[["date", "close"]].copy()
        ret[alias] = ret["close"].pct_change()
        frames.append(ret[["date", alias]].dropna())
    merged = frames[0].merge(frames[1], on="date", how="inner").merge(frames[2], on="date", how="inner")
    return merged.replace([np.inf, -np.inf], np.nan).dropna()


def _single_return_frame(alias: str, prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices[["date", "close"]].copy()
    frame[alias] = frame["close"].pct_change()
    return frame[["date", alias]].replace([np.inf, -np.inf], np.nan).dropna()


def _recent_move_bps(prices: pd.DataFrame, days: int = 5) -> float:
    if len(prices) < 2:
        return 0.0
    window = prices.tail(min(days + 1, len(prices)))
    first = float(window["close"].iloc[0])
    last = float(window["close"].iloc[-1])
    return float((last / first - 1.0) * 10_000) if first > 0 else 0.0


def _peer_cluster(correlation: float, peer_move_bps: float, target_move_bps: float) -> str:
    same_direction = np.sign(peer_move_bps) == np.sign(target_move_bps)
    if correlation >= 0.70 and same_direction:
        return "tight sympathy"
    if correlation >= 0.50:
        return "related sector peer"
    if correlation <= 0.10:
        return "divergent"
    return "loose peer"


def _impact_signal(correlation: float, peer_move_bps: float, side: str) -> str:
    adverse = (side == "buy" and peer_move_bps > 0) or (side == "sell" and peer_move_bps < 0)
    if correlation >= 0.65 and adverse:
        return "crowded adverse flow"
    if correlation >= 0.65:
        return "supportive peer flow"
    if adverse:
        return "weak adverse confirmation"
    return "low peer pressure"


def _peer_urgency_recommendation(
    request: ExecutionRequest,
    crowding_score: float,
    median_peer_move_bps: float,
    target_move_bps: float,
    beta_risk: BetaRiskReport,
) -> tuple[str, str]:
    adverse_peer_move = (request.side == "buy" and median_peer_move_bps > 0) or (
        request.side == "sell" and median_peer_move_bps < 0
    )
    adverse_target_move = (request.side == "buy" and target_move_bps > 0) or (
        request.side == "sell" and target_move_bps < 0
    )
    systematic_heavy = beta_risk.systematic_risk_bps + beta_risk.sector_risk_bps > beta_risk.idiosyncratic_risk_bps
    if crowding_score >= 0.55 and (adverse_peer_move or adverse_target_move or systematic_heavy):
        return (
            "Faster / front-loaded",
            "Peer flow is correlated enough that waiting increases sector/systematic timing risk; "
            "prefer IS or higher-urgency VWAP unless spread/impact is prohibitive.",
        )
    if crowding_score <= 0.25 and not adverse_target_move:
        return (
            "Slower / liquidity-seeking",
            "Peer confirmation is weak and the recent move is not adverse, so a slower VWAP/TWAP style can "
            "reduce impact while accepting timing risk.",
        )
    return (
        "Balanced VWAP / capped POV",
        "Peer evidence is mixed; stay near the 21-day volume curve or use strict POV to avoid overpaying impact.",
    )


def _duration_fraction(request: ExecutionRequest) -> float:
    start = request.start_time.hour * 60 + request.start_time.minute
    end = request.end_time.hour * 60 + request.end_time.minute
    return max(1, end - start) / 390.0


def _vol_bps(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) * 10_000)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _empty_report(
    request: ExecutionRequest,
    mapping: EtfMapping,
    lookback_days: int,
    warnings: list[str],
) -> BetaRiskReport:
    return BetaRiskReport(
        ticker=request.ticker,
        market_etf=MARKET_ETF,
        sector_etf=mapping.sector_etf,
        sector_label=mapping.sector_label,
        mapping_confidence=mapping.confidence,
        mapping_reason=mapping.reason,
        lookback_days=lookback_days,
        observation_count=0,
        beta_market=0.0,
        beta_sector=0.0,
        correlation_market=0.0,
        correlation_sector=0.0,
        r_squared=0.0,
        ticker_daily_vol_bps=0.0,
        market_daily_vol_bps=0.0,
        sector_daily_vol_bps=0.0,
        systematic_risk_bps=0.0,
        sector_risk_bps=0.0,
        idiosyncratic_risk_bps=0.0,
        total_timing_risk_bps=0.0,
        systematic_cost_bps=0.0,
        idiosyncratic_cost_bps=0.0,
        warnings=warnings,
    )


def _empty_peer_report(
    beta_risk: BetaRiskReport,
    candidate_count: int,
    warnings: list[str],
) -> PeerStockAnalysisReport:
    return PeerStockAnalysisReport(
        sector_etf=beta_risk.sector_etf,
        candidate_count=candidate_count,
        analyzed_count=0,
        average_peer_correlation=0.0,
        crowding_score=0.0,
        median_peer_move_bps=0.0,
        target_recent_move_bps=0.0,
        urgency_recommendation="Unavailable",
        rationale="Peer analysis did not have enough live daily history.",
        market_impact_note="Peer flow could not be estimated.",
        warnings=warnings,
    )
