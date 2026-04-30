from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from execlab.pretrade import build_pretrade_analytics, fetch_lookback_sessions, fit_expected_cost_model
from execlab.risk import build_beta_risk_report, build_peer_stock_analysis, map_ticker_to_etf
from execlab.schemas import ExecutionRequest
from tests.conftest import FakeMarketDataClient, synthetic_bars


def test_lookback_curves_align_to_market_time_buckets() -> None:
    client = FakeMarketDataClient()
    lookback, warnings = fetch_lookback_sessions(client, "NVDA", date(2026, 4, 20), "5m", 5)
    request = ExecutionRequest(ticker="NVDA", trade_date=date(2026, 4, 20), quantity=25_000)
    report = build_pretrade_analytics(request, synthetic_bars(), lookback, requested_sessions=5, warnings=warnings)

    assert report.lookback_sessions == 5
    assert len(report.curve) == 78
    assert report.adv_shares > 0
    assert report.avg_spread_proxy_bps > 0
    assert report.avg_volatility_bps >= 0


def test_expected_cost_regression_returns_finite_outputs() -> None:
    client = FakeMarketDataClient()
    lookback, warnings = fetch_lookback_sessions(client, "NVDA", date(2026, 4, 20), "5m", 5)
    request = ExecutionRequest(ticker="NVDA", trade_date=date(2026, 4, 20), quantity=25_000)
    pretrade = build_pretrade_analytics(request, synthetic_bars(), lookback, requested_sessions=5, warnings=warnings)
    report = fit_expected_cost_model(request, pretrade, lookback, seed=123)

    assert np.isfinite(report.expected_cost_bps)
    assert report.observation_count > 0
    assert len(report.coefficients) >= 5
    assert len(report.cost_breakdown) >= 5
    assert len(report.through_day_cost) == len(pretrade.curve)


def test_beta_risk_maps_sector_etf_and_returns_finite_risk() -> None:
    client = FakeMarketDataClient()
    request = ExecutionRequest(ticker="NVDA", trade_date=date(2026, 4, 20), quantity=25_000)
    mapping = map_ticker_to_etf("NVDA")
    report = build_beta_risk_report(request, client, lookback_days=63)

    assert mapping.sector_etf == "XLK"
    assert report.sector_etf == "XLK"
    assert report.observation_count >= 20
    assert np.isfinite(report.beta_market)
    assert np.isfinite(report.idiosyncratic_risk_bps)
    assert report.total_timing_risk_bps >= 0
    assert len(report.index_comparison) == 78


def test_peer_stock_analysis_returns_clusters_and_urgency() -> None:
    client = FakeMarketDataClient()
    request = ExecutionRequest(ticker="NVDA", trade_date=date(2026, 4, 20), quantity=25_000)
    beta = build_beta_risk_report(request, client, lookback_days=63)
    peer = build_peer_stock_analysis(request, client, beta, lookback_days=63, max_candidates=6)

    assert peer.candidate_count > 0
    assert peer.analyzed_count > 0
    assert peer.urgency_recommendation in {
        "Faster / front-loaded",
        "Slower / liquidity-seeking",
        "Balanced VWAP / capped POV",
    }
    assert peer.peers
